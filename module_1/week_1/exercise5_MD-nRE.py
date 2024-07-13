'''
- Mean Difference of n_th Root Error: là một kỹ thuật thông dụng trong các
ứng dụng như phát hiện và theo dõi đối tượng. Ngoài ra, phương pháp này
cũng có thể được áp dụng cho các bài toán hồi quy khác.

- Behavior of MD_nRE: MD_nRE cho thấy các giá trị thay đổi, giảm dần khi độ
lớn của yi (và yˆi) tăng lên. Điều này chỉ ra rằng MD_nRE, không giống như MAE,
nhạy cảm với tỷ lệ của dữ liệu. Đối với các giá trị nhỏ hơn của yi và yˆi,
việc chuyển đổi căn bậc có ảnh hưởng đáng kể hơn đến việc tính toán loss, dẫn
đến giá trị MD_nRE cao hơn. Khi các giá trị tăng lên, sự khác biệt tương đối
giữa các căn giảm xuống, dẫn đến giá trị MD_nRE thấp hơn.

- Một số ưu điểm khi sử dụng MD_nRE làm hàm loss cho các bài toán regression:

    • Robustness to outliers: Việc lấy căn bậc n trước khi tính loss, đặc biệt
    cho các trường hợp loss lớn (do outliers) có thể giúp việc optimization của
    model trở nên dễ dàng và mượt mà hơn.

    • Improved convergence: Quá trình training nhanh hơn, hội tụ và ổn định hơn
    khi áp dụng kỹ thuật này.

    • Application domains: Phương pháp này đặc biệt phù hợp trong các lĩnh vực mà
    độ lớn của sai số đóng vai trò quan trọng, như tài chính, nơi các sai số nhỏ
    có thể có tác động tài chính đáng kể.

'''

import math


def md_nre_single_sample(y, y_hat, n, p):
    md_nre = (y ** (1 / n) - y_hat ** (1 / n)) ** p
    print(f"md_nre_single_sample (y={y}, y_hat={
          y_hat}, n={n}, p={p}): {md_nre}")
    return md_nre


if __name__ == "__main__":
    number_samples = int(input("Enter number of samples:"))
    n = int(input("Enter n: "))
    p = int(input("Enter p: "))
    md_nre = 0

    for i in range(number_samples):
        y = float(input(f"Enter y{i+1}: "))
        y_hat = float(input(f"Enter y_hat{i+1}: "))
        md_nre += md_nre_single_sample(y, y_hat, n, p)

    md_nre /= number_samples
    print(f"md_nre: {md_nre}")
