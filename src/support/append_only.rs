use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::{mem, ptr};

struct Block<T> {
    data: T,
    next: AtomicPtr<Block<T>>,
}

impl<T> Block<T> {
    fn new(data: T) -> Self {
        Self {
            data,
            next: AtomicPtr::new(ptr::null_mut()),
        }
    }
}

struct Head<T>(Block<T>);

impl<T> Head<T> {
    fn new(data: T) -> Self {
        Self(Block::new(data))
    }

    // returns new length
    fn push(&self, data: T) -> usize {
        let new_block_ptr = Box::leak(Box::new(Block::new(data)));

        let mut len = 1;
        let mut next_ptr = &self.0.next;
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

        // the first block is owned directly and is
        // automatically dropped at the end of this scope
        let mut next = self.0.next.load(Ordering::Relaxed);
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
    pub fn new(data: T) -> Self {
        Self(Arc::new(Head::new(data)))
    }

    // returns new length
    pub fn push(&self, data: T) -> usize {
        self.0.push(data)
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            block: &raw const self.0.as_ref().0,
            _phantom: PhantomData,
        }
    }
}
