"""Mini NPU Simulator.

Run on Windows with: py -X utf8 main.py
Uses only the Python standard library.
"""

import json
import math
import time
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / "data.json"
SIZES = [3, 5, 13, 25]
EPSILON = 1e-9


def normalize_label(label):
    """Normalize accepted class labels to Cross, X, or UNDECIDED."""
    value = str(label).strip().lower().replace(" ", "")
    if value in {"+", "cross", "plus"}:
        return "Cross"
    if value in {"x", "×"}:
        return "X"
    if value in {"undecided", "판정불가", "tie", "동점"}:
        return "UNDECIDED"
    raise ValueError(
        f"알 수 없는 라벨입니다: {label!r}. Cross, +, plus, X, ×, UNDECIDED 중 하나를 사용하세요."
    )


def validate_matrix(matrix, size, name="행렬"):
    """Validate an N x N numeric matrix and return values as finite floats."""
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError(f"{name}: {size}행 행렬이어야 합니다.")

    result = []
    for row_number, row in enumerate(matrix, start=1):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(
                f"{name}: {row_number}번째 행에는 숫자 {size}개가 있어야 합니다."
            )
        converted_row = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}: 모든 원소는 숫자여야 합니다. 문자열과 True/False는 사용할 수 없습니다.")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name}: NaN 또는 무한대가 아닌 유한한 숫자만 사용할 수 있습니다.")
            converted_row.append(number)
        result.append(converted_row)
    return result


def mac(pattern, filter_matrix):
    """Multiply matching matrix cells and accumulate their products (MAC)."""
    if len(pattern) != len(filter_matrix):
        raise ValueError("입력 패턴과 필터의 행 개수가 서로 다릅니다.")

    total = 0.0
    for row in range(len(pattern)):
        if len(pattern[row]) != len(filter_matrix[row]):
            raise ValueError("입력 패턴과 필터의 행 길이가 서로 다릅니다.")
        for col in range(len(pattern[row])):
            total += pattern[row][col] * filter_matrix[row][col]
    return total


def generate_pattern(size, pattern_type):
    """Generate an N x N Cross or X matrix containing 1.0 and 0.0."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("패턴 크기 N은 1 이상의 정수여야 합니다.")

    kind = normalize_label(pattern_type)
    if kind == "UNDECIDED":
        raise ValueError("패턴 종류는 Cross 또는 X여야 합니다.")

    center = size // 2
    matrix = []
    for row in range(size):
        values = []
        for col in range(size):
            if kind == "Cross":
                value = 1.0 if row == center or col == center else 0.0
            else:
                value = 1.0 if row == col or row + col == size - 1 else 0.0
            values.append(value)
        matrix.append(values)
    return matrix


def print_matrix(matrix):
    """Print a matrix in a compact, readable form."""
    for row in matrix:
        print(" ".join(f"{value:.0f}" for value in row))


def flatten_matrix(matrix):
    """Convert a 2D matrix into a row-major 1D list."""
    return [value for row in matrix for value in row]


def mac_flat(pattern_flat, filter_flat):
    """Run MAC using flattened 1D lists."""
    if len(pattern_flat) != len(filter_flat):
        raise ValueError("1차원 입력 패턴과 필터의 길이가 서로 다릅니다.")
    total = 0.0
    for index in range(len(pattern_flat)):
        total += pattern_flat[index] * filter_flat[index]
    return total


def matrix_norm(matrix):
    """Return the Euclidean (L2) norm of a matrix treated as one vector."""
    total = 0.0
    for row in matrix:
        for value in row:
            total += value * value
    return math.sqrt(total)


def normalized_mac(pattern, filter_matrix):
    """Return cosine-similarity-style normalized MAC score."""
    raw_score = mac(pattern, filter_matrix)
    denominator = matrix_norm(pattern) * matrix_norm(filter_matrix)
    if denominator == 0:
        return 0.0
    return raw_score / denominator


def classify(score_cross, score_x):
    """Choose the higher score; values within EPSILON are treated as a tie."""
    if abs(score_cross - score_x) < EPSILON:
        return "UNDECIDED"
    return "Cross" if score_cross > score_x else "X"


def read_matrix(size, title):
    """Read a square matrix from the console, retrying invalid rows."""
    print(f"\n{title} ({size}x{size})")
    example = " ".join(["0"] * size)
    print(f"각 행에 숫자 {size}개를 공백으로 구분해 입력하세요. 예: {example}")
    matrix = []
    for row_number in range(1, size + 1):
        while True:
            raw = input(f"{row_number}행: ").strip()
            try:
                values = [float(item) for item in raw.split()]
                if len(values) != size:
                    print(f"숫자를 정확히 {size}개 입력하세요. 예: {example}")
                    continue
                if not all(math.isfinite(value) for value in values):
                    print("NaN 또는 무한대가 아닌 유한한 숫자를 입력하세요.")
                    continue
                matrix.append(values)
                break
            except ValueError:
                print(f"숫자만 입력하세요. 예: {example}")
    return matrix


def show_scores(pattern, filter_cross, filter_x, undecided_text="UNDECIDED"):
    """Print raw and normalized scores; return the internal class label."""
    raw_cross = mac(pattern, filter_cross)
    raw_x = mac(pattern, filter_x)
    score_cross = normalized_mac(pattern, filter_cross)
    score_x = normalized_mac(pattern, filter_x)

    print(f"Cross MAC 합계: {raw_cross:.6f}")
    print(f"X MAC 합계: {raw_x:.6f}")
    print(f"Cross 정규화 점수: {score_cross:.6f}")
    print(f"X 정규화 점수: {score_x:.6f}")

    result = classify(score_cross, score_x)
    display_result = undecided_text if result == "UNDECIDED" else result
    print(f"판정 결과: {display_result}")
    return result


def measure_performance():
    """Measure average MAC time (excluding input, generation, and printing)."""
    print("\n========== 성능 측정 ==========")
    repetitions = 100
    print(f"크기별 반복 횟수: {repetitions}회")
    print("크기    평균 시간(ms)   연산 횟수(N²)")
    for size in SIZES:
        pattern = generate_pattern(size, "Cross")
        filter_matrix = generate_pattern(size, "X")
        start = time.perf_counter()
        for _ in range(repetitions):
            mac(pattern, filter_matrix)
        elapsed = time.perf_counter() - start
        average_ms = (elapsed / repetitions) * 1000
        operations = size * size
        print(f"{size}x{size:<5} {average_ms:>12.6f} {operations:>16}")
    print("측정 범위: MAC 계산만 포함하며 입력·파일 읽기·패턴 생성·콘솔 출력은 제외합니다.")


def measure_optimization_performance():
    """Compare 2D MAC with flattened 1D MAC using identical inputs and repeats."""
    print("\n========== 보너스 1: 2차원 / 1차원 성능 비교 ==========")
    repetitions = 1000
    rounds = 5
    print(f"동일 입력 · 크기별 {repetitions}회 반복 · {rounds}회 측정 평균")
    print("1차원 방식은 각 라운드의 배열 변환 시간도 포함합니다.")
    print("크기    2D 총시간(ms)   1D 변환+계산(ms)   차이")

    for size in SIZES:
        pattern = generate_pattern(size, "Cross")
        filter_matrix = generate_pattern(size, "X")
        expected = mac(pattern, filter_matrix)
        flat_expected = mac_flat(flatten_matrix(pattern), flatten_matrix(filter_matrix))
        if not math.isclose(expected, flat_expected, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"{size}x{size}: 2D와 1D MAC 계산 결과가 다릅니다.")

        total_2d_seconds = 0.0
        total_1d_seconds = 0.0
        for round_number in range(rounds):
            conversion_start = time.perf_counter()
            pattern_flat = flatten_matrix(pattern)
            filter_flat = flatten_matrix(filter_matrix)
            conversion_seconds = time.perf_counter() - conversion_start

            if round_number % 2 == 0:
                start = time.perf_counter()
                for _ in range(repetitions):
                    mac(pattern, filter_matrix)
                two_d_seconds = time.perf_counter() - start
                start = time.perf_counter()
                for _ in range(repetitions):
                    mac_flat(pattern_flat, filter_flat)
                one_d_seconds = time.perf_counter() - start
            else:
                start = time.perf_counter()
                for _ in range(repetitions):
                    mac_flat(pattern_flat, filter_flat)
                one_d_seconds = time.perf_counter() - start
                start = time.perf_counter()
                for _ in range(repetitions):
                    mac(pattern, filter_matrix)
                two_d_seconds = time.perf_counter() - start

            total_2d_seconds += two_d_seconds
            total_1d_seconds += conversion_seconds + one_d_seconds

        average_2d_ms = (total_2d_seconds / rounds) * 1000
        average_1d_ms = (total_1d_seconds / rounds) * 1000
        if average_1d_ms < average_2d_ms:
            difference = (average_2d_ms - average_1d_ms) / average_2d_ms * 100
            comparison = f"1D {difference:.1f}% 빠름"
        elif average_1d_ms > average_2d_ms:
            difference = (average_1d_ms - average_2d_ms) / average_2d_ms * 100
            comparison = f"1D {difference:.1f}% 느림"
        else:
            comparison = "차이 거의 없음"
        print(
            f"{size}x{size:<5} {average_2d_ms:>13.6f}"
            f" {average_1d_ms:>18.6f}   {comparison}"
        )

    print("참고: 실행 환경과 다른 프로그램의 동작에 따라 측정값은 달라질 수 있습니다.")


def user_mode():
    print("\n========== 사용자 입력 모드 ==========")
    size = 3
    filter_cross = read_matrix(size, "Cross 필터 입력")
    filter_x = read_matrix(size, "X 필터 입력")

    print("\n분류할 패턴 입력 방법")
    print("1. 직접 입력")
    print("2. 패턴 자동 생성기 사용")
    while True:
        pattern_choice = input("선택 (1 또는 2): ").strip()
        if pattern_choice in {"1", "2"}:
            break
        print("1 또는 2를 입력하세요.")

    if pattern_choice == "1":
        pattern = read_matrix(size, "분류할 패턴 입력")
    else:
        print("\n자동 생성할 패턴 종류")
        print("1. Cross")
        print("2. X")
        while True:
            kind_choice = input("선택 (1 또는 2): ").strip()
            if kind_choice in {"1", "2"}:
                break
            print("1 또는 2를 입력하세요.")
        kind = "Cross" if kind_choice == "1" else "X"
        pattern = generate_pattern(size, kind)
        print(f"\n자동 생성된 {kind} 패턴:")
        print_matrix(pattern)

    print("\n========== 분석 결과 ==========")
    show_scores(pattern, filter_cross, filter_x, undecided_text="판정 불가")
    # The user-mode timing summary appears after the classification result.
    measure_performance()


def pattern_generator_mode():
    """Ask for N and display generated Cross and X matrices."""
    print("\n========== 보너스 2: 패턴 자동 생성기 ==========")
    while True:
        raw_size = input("패턴 크기 N을 입력하세요 (1 이상의 정수): ").strip()
        try:
            size = int(raw_size)
            if size < 1:
                print("N은 1 이상의 정수여야 합니다.")
                continue
            break
        except ValueError:
            print("정수만 입력하세요. 예: 5")

    cross_pattern = generate_pattern(size, "Cross")
    x_pattern = generate_pattern(size, "X")
    print(f"\n[Cross 패턴 {size}x{size}]")
    print_matrix(cross_pattern)
    print(f"\n[X 패턴 {size}x{size}]")
    print_matrix(x_pattern)
    print("\n생성된 패턴은 사용자 입력 모드와 성능 비교에서도 사용됩니다.")


def load_json():
    """Load the JSON data file stored beside this main.py file."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE.name} 파일을 찾을 수 없습니다. main.py와 같은 폴더에 넣으세요."
        )
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_filter_matrix(size_filters, wanted_label, size):
    """Find a filter by normalized key, accepting Cross/X label aliases."""
    if not isinstance(size_filters, dict):
        raise ValueError(f"size_{size} 필터 항목은 객체(dict)여야 합니다.")
    for key, matrix in size_filters.items():
        try:
            normalized_key = normalize_label(key)
        except ValueError:
            # Ignore unrelated metadata keys, if any.
            continue
        if normalized_key == wanted_label:
            return validate_matrix(matrix, size, f"size_{size} {wanted_label} 필터")
    raise ValueError(f"size_{size}에 {wanted_label} 필터가 없습니다. cross/x 키를 확인하세요.")


def parse_pattern_size(pattern_name):
    """Extract N from the required size_N_idx pattern key."""
    parts = pattern_name.split("_")
    if (
        len(parts) != 3
        or parts[0] != "size"
        or not parts[1].isdigit()
        or not parts[2].isdigit()
    ):
        raise ValueError("패턴 이름은 size_N_idx 형식이어야 합니다. 예: size_5_1")
    size = int(parts[1])
    if size < 1:
        raise ValueError("패턴 이름의 N은 1 이상의 정수여야 합니다.")
    return size


def json_mode():
    print("\n========== JSON 분석 모드 ==========")
    data = load_json()
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 구조는 객체여야 합니다.")
    filters = data.get("filters")
    patterns = data.get("patterns")
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        raise ValueError("JSON 최상위에 filters와 patterns 객체가 모두 있어야 합니다.")

    total = 0
    passed = 0
    failures = []
    failure_counts = {
        "데이터 오류": 0,
        "동점 판정(UNDECIDED)": 0,
        "일반 판정 불일치": 0,
    }

    for pattern_name, item in patterns.items():
        total += 1
        try:
            if not isinstance(pattern_name, str):
                raise ValueError("패턴 이름은 문자열이어야 합니다.")
            if not isinstance(item, dict):
                raise ValueError("패턴 항목은 객체(dict)여야 합니다.")
            if "input" not in item or "expected" not in item:
                raise ValueError("input 또는 expected 항목이 없습니다.")

            size = parse_pattern_size(pattern_name)
            filter_key = f"size_{size}"
            if filter_key not in filters:
                raise ValueError(f"{filter_key} 필터가 JSON filters에 없습니다.")
            pattern = validate_matrix(item["input"], size, f"{pattern_name} 입력")
            size_filters = filters[filter_key]
            filter_cross = get_filter_matrix(size_filters, "Cross", size)
            filter_x = get_filter_matrix(size_filters, "X", size)
            expected = normalize_label(item["expected"])
        except (KeyError, TypeError, ValueError) as error:
            failure_counts["데이터 오류"] += 1
            failures.append((pattern_name, "데이터 오류", str(error)))
            print(f"\n[{pattern_name}] 데이터 오류: {error}")
            continue

        raw_cross = mac(pattern, filter_cross)
        raw_x = mac(pattern, filter_x)
        score_cross = normalized_mac(pattern, filter_cross)
        score_x = normalized_mac(pattern, filter_x)
        result = classify(score_cross, score_x)
        is_pass = result == expected

        if is_pass:
            passed += 1
        else:
            if result == "UNDECIDED":
                reason_type = "동점 판정(UNDECIDED)"
            else:
                reason_type = "일반 판정 불일치"
            failure_counts[reason_type] += 1
            failures.append((pattern_name, reason_type, f"기대값 {expected}, 실제 판정 {result}"))

        print(f"\n[{pattern_name}]")
        print(f"Cross MAC 합계: {raw_cross:.6f}")
        print(f"X MAC 합계: {raw_x:.6f}")
        print(f"Cross 정규화 점수: {score_cross:.6f}")
        print(f"X 정규화 점수: {score_x:.6f}")
        print(f"판정 결과: {result}")
        print(f"기대 결과: {expected}")
        print(f"테스트 결과: {'PASS' if is_pass else 'FAIL'}")

    print("\n========== 결과 리포트 ==========")
    print(f"전체 테스트: {total}")
    print(f"통과: {passed}")
    print(f"실패: {total - passed}")
    print("\n실패 원인별 집계:")
    for reason, count in failure_counts.items():
        print(f"- {reason}: {count}")

    if failures:
        print("\n실패 케이스:")
        for name, reason_type, detail in failures:
            print(f"- {name} [{reason_type}]: {detail}")
    else:
        print("모든 테스트를 통과했습니다.")

    measure_performance()


def main():
    print("================================")
    print("       Mini NPU Simulator")
    print("================================")
    print("1. 사용자 입력 모드")
    print("2. JSON 데이터 분석 모드")
    print("3. 패턴 자동 생성기 (보너스)")
    print("4. 2차원 / 1차원 최적화 성능 비교 (보너스)")
    choice = input("\n실행할 모드를 선택하세요 (1~4): ").strip()

    try:
        if choice == "1":
            user_mode()
        elif choice == "2":
            json_mode()
        elif choice == "3":
            pattern_generator_mode()
        elif choice == "4":
            measure_optimization_performance()
        else:
            print("1~4 중 하나를 입력하세요.")
    except (FileNotFoundError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        print(f"\n오류: {error}")


if __name__ == "__main__":
    main()
