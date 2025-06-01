int factorial(int n) {
    int result = 1;
    int unused_var = 0;
    
    for(int i = 1; i <= n; i = i + 1) {
        result = result * i;
        unused_var = unused_var + 1;
    }
    
    return result + 0;
}

int compute_sum(int a, int b) {
    int x = a + 0;
    int y = b * 1;
    int z = x - x;
    int dead_var = 100;
    
    return x + y + z;
}

int optimize_expressions() {
    int a = 5 + 3;
    int b = a * 2;
    int c = b - b;
    int d = c + 10;
    int e = d * 0;
    int f = e + 15;
    
    return f;
}

int main() {
    int x = 5;
    int y = factorial(x) * 1;
    int z = compute_sum(10, 20);
    
    int unused_result = optimize_expressions() + 0;
    
    return y + z;
}