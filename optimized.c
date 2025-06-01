int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i = i + 1) {
        result = result * i;
    }
    return result;
}
int compute_sum(int a, int b) {
    int x = a;
    int y = b;
    int z = 0;
    return x + y + z;
}
int optimize_expressions() {
    int a = 8;
    int b = a * 2;
    int c = 0;
    int d = c + 10;
    int e = 0;
    int f = e + 15;
    return f;
}
int main() {
    int x = 5;
    int y = factorial(x);
    int z = compute_sum(10, 20);
    return y + z;
}