'''
Ưu điểm của F1-Score so với Accuracy:

- Cân bằng: F1-score cung cấp một sự cân bằng tốt hơn giữa độ chính xác và độ nhạy,
  đặc biệt trong các tình huống mà phân phối lớp không đồng đều.

- Không nhạy cảm với mất cân bằng lớp: Trong các tập dữ liệu mà một lớp xuất hiện
  nhiều hơn đáng kể so với lớp khác, độ chính xác có thể gây hiểu lầm. F1-score
  cung cấp thông tin hữu ích hơn trong những trường hợp này.

- Xử lý tốt các trường hợp dương tính giả và âm tính giả: F1-score xem xét cả hai
  yếu tố dương tính giả và âm tính giả, cung cấp một cái nhìn toàn diện hơn về hiệu
  suất của bộ phân loại.
'''


def calculate_f1_score(tp, fp, fn):
    '''
    This function calculates the precision, recall, f1 score
    Args:
        tp: true positive
        fp: false positive
        fn: false negative
    Returns:
        precision, recall, f1_score
    Constraints:
        tp, fp, fn must be integers and greater than zero.
    '''
    if not isinstance(tp, int) or not isinstance(fp, int) or not isinstance(fn, int):
        if not isinstance(tp, int):
            print("tp must be int")
        if not isinstance(fp, int):
            print("fp must be int")
        if not isinstance(fn, int):
            print("fn must be int")
        return
    if (tp < 0) or (fp < 0) or (fn < 0):
        print("tp and fp and fn must be greater than zero")
        return

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1_score: {f1_score}")

    return f1_score


if __name__ == "__main__":
    calculate_f1_score('a', 10, 20)
    calculate_f1_score(10, 'b', 20)
    calculate_f1_score(10, 20, 'c')
    calculate_f1_score(-10, 20, 30)
    calculate_f1_score(10, -20, 30)
    calculate_f1_score(10, 20, -30)
    calculate_f1_score(10, 20, 30)
