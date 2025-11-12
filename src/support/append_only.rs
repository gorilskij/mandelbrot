use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::{mem, ptr};

type Link<T> = AtomicPtr<Block<T>>;

struct Block<T> {
    data: T,
    next: Link<T>,
}

impl<T> Block<T> {
    fn new(data: T) -> Self {
        Self {
            data,
            next: Default::default(),
        }
    }
}

struct Head<T>(Link<T>);

impl<T> Head<T> {
    fn new() -> Self {
        Self(Default::default())
    }

    fn push_front(&self, data: T) {
        let mut current_first_block = self.0.load(Ordering::Acquire);
        let new_block_ptr = Box::leak(Box::new(Block {
            data,
            next: AtomicPtr::new(current_first_block),
        }));

        while let Err(actual_first_block) = self.0.compare_exchange(
            current_first_block,
            new_block_ptr,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            current_first_block = actual_first_block;
            new_block_ptr
                .next
                .store(actual_first_block, Ordering::Release);
        }
    }

    // returns new length
    fn push_back(&self, data: T) -> usize {
        let new_block_ptr = Box::leak(Box::new(Block::new(data)));

        let mut len = 0;
        let mut next_ptr = &self.0;
        while let Err(next) = next_ptr.compare_exchange(
            ptr::null_mut(),
            new_block_ptr,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            let offset = mem::offset_of!(Block<T>, next);

            // TODO: SAFETY
            next_ptr = unsafe { &*(next.offset(offset as isize) as *mut AtomicPtr<_>) };

            len += 1;
        }

        len
    }
}

impl<T> Drop for Head<T> {
    fn drop(&mut self) {
        // NOTE: Relaxed ordering is used here because this code
        // only runs when all threads but one have dropped their
        // Arc handles, this provides Acquire/Release synchronization
        // ensuring all their writes are visible, otherwise this
        // code would need to use Acquire

        let mut next = self.0.load(Ordering::Relaxed);
        while !next.is_null() {
            // SAFETY: next is not null, this code can only be run from a single thread
            let block = unsafe { Box::from_raw(next) };
            next = block.next.load(Ordering::Relaxed);
        }
    }
}

pub struct Iter<'a, T> {
    block: *const Block<T>,
    _phantom: PhantomData<&'a Head<T>>,
}

impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.block.is_null() {
            return None;
        }

        // SAFETY: while this Iter holds the lifetime 'a,
        //   it prevents the associated ListHandle from
        //   being dropped, which in turn holds a handle
        //   to the Arc preventing the List from running
        //   its Drop code (the underlying blocks can't
        //   be dropped during 'a)
        let block = unsafe { &*self.block };

        self.block = block.next.load(Ordering::Acquire);

        Some(&block.data)
    }
}

// this is a handle that can cheaply be cloned into other handles for other threads
pub struct List<T>(Arc<Head<T>>);

impl<T> Clone for List<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> List<T> {
    pub fn new() -> Self {
        Self(Arc::new(Head::new()))
    }

    pub fn push_front(&self, data: T) {
        self.0.push_front(data)
    }

    // returns new length
    pub fn push_back(&self, data: T) -> usize {
        self.0.push_back(data)
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            block: self.0.0.load(Ordering::Acquire),
            _phantom: PhantomData,
        }
    }
}
