# Khai báo hàm và đạo hàm
def f(x):
    return x**2 + 2*x + 5  # Hàm f(x)

def f_prime(x):
    return 2*x + 2  # Đạo hàm f'(x)

# Gradient Descent
def gradient_descent(learning_rate, iterations, start_point):
    x = start_point  # Điểm khởi tạo
    for i in range(iterations):
        grad = f_prime(x)  # Tính gradient tại x
        x = x - learning_rate * grad  # Cập nhật x
        print(f"Iteration {i+1}: x = {x}, f(x) = {f(x)}")
    return x

# Tham số thuật toán
learning_rate = 0.1  # Tốc độ học
iterations = 50  # Số lần lặp
start_point = 5  # Điểm khởi tạo

# Chạy thuật toán Gradient Descent
optimal_x = gradient_descent(learning_rate, iterations, start_point)
print(f"x = {optimal_x}, f(x) = {f(optimal_x)}")
