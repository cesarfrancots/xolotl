use stack::Stack;

#[test]
fn peek_on_an_empty_stack_is_none() {
    let s: Stack<i32> = Stack::new();
    assert_eq!(s.peek(), None);
}

#[test]
fn peek_returns_the_most_recent_push() {
    let mut s = Stack::new();
    s.push(10);
    assert_eq!(s.peek(), Some(&10));
    s.push(20);
    s.push(30);
    assert_eq!(s.peek(), Some(&30));
}

#[test]
fn peek_does_not_remove_the_top() {
    let mut s = Stack::new();
    s.push(1);
    s.push(2);
    s.push(3);
    assert_eq!(s.peek(), Some(&3));
    assert_eq!(s.peek(), Some(&3));
    assert_eq!(s.len(), 3);
    assert_eq!(s.pop(), Some(3));
    assert_eq!(s.peek(), Some(&2));
    assert_eq!(s.pop(), Some(2));
    assert_eq!(s.pop(), Some(1));
    assert_eq!(s.peek(), None);
}

#[test]
fn peek_takes_a_shared_reference_and_borrows_the_item() {
    fn top_of(s: &Stack<String>) -> Option<&String> {
        s.peek()
    }

    let mut s = Stack::new();
    s.push("a".to_string());
    s.push("b".to_string());
    let top: Option<&String> = top_of(&s);
    assert_eq!(top.map(|t| t.as_str()), Some("b"));
    assert_eq!(s.len(), 2);
}
