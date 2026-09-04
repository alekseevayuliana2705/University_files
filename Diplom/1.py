def hamming_error_correction(message):
    # Преобразуем строку в список для удобной модификации
    bits = list(message)

    # Проверим контрольные биты
    error_position = 0
    length = len(bits)
    
    # Проверяем биты на позициях степеней двойки (1, 2, 4, 8, ...).
    for i in range(length):
        # Позиции контрольных битов: 1, 2, 4, 8, 16, 32, ...
        if (i + 1) & (i + 1 - 1) == 0:  # Это контрольный бит (позиции 1, 2, 4, 8, ...).
            check_bits = []
            for j in range(i + 1, length, 2 * (i + 1)):
                check_bits.extend(bits[j:min(j + (i + 1), length)])
            parity = sum(int(bit) for bit in check_bits) % 2
            if parity != int(bits[i]):  # если контрольный бит не совпадает с вычисленным.
                error_position ^= i + 1

    if error_position:
        print(f"Ошибка обнаружена на позиции {error_position}")
        bits[error_position - 1] = str(1 - int(bits[error_position - 1]))  # исправляем ошибку
        print(f"Исправленное сообщение: {''.join(bits)}")
    else:
        print("Ошибок не найдено")

    # Убираем контрольные биты.
    info_bits = [bits[i] for i in range(len(bits)) if (i + 1) & (i + 1 - 1) != 0]
    
    # Возвращаем результат в десятичной форме.
    info_bits_str = ''.join(info_bits)
    return int(info_bits_str, 2)

# Вводим сообщение
message = "111000111010010"
result = hamming_error_correction(message)
print(result)