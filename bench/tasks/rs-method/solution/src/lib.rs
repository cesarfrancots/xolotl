//! A minimal last-in, first-out stack.

/// A last-in, first-out stack of `T`.
#[derive(Debug)]
pub struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    /// An empty stack.
    pub fn new() -> Self {
        Stack { items: Vec::new() }
    }

    /// Puts `item` on top of the stack.
    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }

    /// Removes and returns the top item, or `None` if the stack is empty.
    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    /// The top item without removing it, or `None` if the stack is empty.
    pub fn peek(&self) -> Option<&T> {
        self.items.last()
    }

    /// Number of items on the stack.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// `true` when the stack holds no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pop_returns_items_in_reverse_order() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        s.push(3);
        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(1));
        assert_eq!(s.pop(), None);
    }

    #[test]
    fn pop_on_an_empty_stack_is_none() {
        let mut s: Stack<i32> = Stack::new();
        assert_eq!(s.pop(), None);
    }

    #[test]
    fn len_and_is_empty_follow_the_contents() {
        let mut s = Stack::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        s.push("a");
        assert!(!s.is_empty());
        assert_eq!(s.len(), 1);
        s.pop();
        assert!(s.is_empty());
    }

    #[test]
    fn peek_shows_the_top_without_removing_it() {
        let mut s = Stack::new();
        assert_eq!(s.peek(), None);
        s.push(1);
        s.push(2);
        assert_eq!(s.peek(), Some(&2));
        assert_eq!(s.len(), 2);
    }
}
