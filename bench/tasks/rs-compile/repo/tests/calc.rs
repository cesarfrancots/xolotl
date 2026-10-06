use calc::{digit_sum, gcd, with_doubles};

#[test]
fn digit_sum_of_zero_is_zero() {
    let got: u32 = digit_sum(0);
    assert_eq!(got, 0);
}

#[test]
fn digit_sum_adds_every_digit() {
    let got: u32 = digit_sum(1234);
    assert_eq!(got, 10);
}

#[test]
fn digit_sum_of_the_largest_u32() {
    let got: u32 = digit_sum(u32::MAX);
    assert_eq!(got, 57);
}

#[test]
fn gcd_of_common_cases() {
    assert_eq!(gcd(12, 18), 6);
    assert_eq!(gcd(17, 5), 1);
    assert_eq!(gcd(0, 9), 9);
    assert_eq!(gcd(0, 0), 0);
}

#[test]
fn with_doubles_appends_the_doubles_after_the_originals() {
    assert_eq!(with_doubles(vec![1, 2, 3]), vec![1, 2, 3, 2, 4, 6]);
}

#[test]
fn with_doubles_of_nothing_is_nothing() {
    assert_eq!(with_doubles(Vec::new()), Vec::<u32>::new());
}
