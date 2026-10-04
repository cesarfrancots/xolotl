//! Small number helpers.

/// Sum of the decimal digits of `n`: `digit_sum(1234)` is `10`.
pub fn digit_sum(n: u32) -> u32 {
    let mut rest = n;
    let mut total: u32 = 0;
    while rest > 0 {
        total += rest % 10;
        rest /= 10;
    }
    total
}

/// Greatest common divisor, with `gcd(0, 0) == 0`.
pub fn gcd(a: u32, b: u32) -> u32 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// `values` followed by the double of each of its elements, in order:
/// `with_doubles(vec![1, 2, 3])` is `[1, 2, 3, 2, 4, 6]`.
pub fn with_doubles(mut values: Vec<u32>) -> Vec<u32> {
    let doubles: Vec<u32> = values.iter().map(|v| v * 2).collect();
    values.extend(doubles);
    values
}
