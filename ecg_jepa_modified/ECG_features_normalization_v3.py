import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def normalize_ecg_features(features):
    """
    Нормализует массив ЭКГ признаков размерности (n_records, n_features, n_timesteps)
    с заменой NaN на среднее значение признака по всем замерам

    Параметры:
    features : ndarray, shape (n_records, n_features, n_timesteps)

    Возвращает:
    normalized : нормализованный массив той же формы без NaN значений
    """
    n_records, n_features, n_timesteps = features.shape
    normalized = np.zeros_like(features)

    # Вычисляем средние значения по каждому признаку (игнорируя NaN)
    feature_means = np.nanmean(features, axis=(0, 2))  # shape (n_features,)

    # 1. Обрабатываем исходные сигналы ЭКГ (каналы 0-7)
    for j in range(8):
        #     # Заменяем NaN на среднее по этому признаку
        #     channel_data = np.where(np.isnan(features[:, j, :]),
        #                           feature_means[j],
        #                           features[:, j, :])

        # np.zeros(num_samples),  # placeholder для r_peaks+
        # hr,+
        # rolling_hr,+
        # st_dev, +
        # qrs_amp, +
        # np.full(num_samples, sdnn),
        # np.full(num_samples, rmssd),
        # wavelet_coeffs

        #     # Z-score нормализация
        #     scaler = StandardScaler()
        normalized[:, j, :] = features[:, j, :]
    #         channel_data.reshape(-1, 1)).reshape(n_records, n_timesteps)

    # 2. Бинарные признаки (r_peaks) - только замена NaN
    j = 8
    normalized[:, j, :] = np.where(
        np.isnan(features[:, j, :]), feature_means[j], features[:, j, :]
    )

    # 3. Признаки с нормальным распределением (heart_rate, rolling_hr)
    for j in [9, 10]:
        channel_data = np.where(
            np.isnan(features[:, j, :]), feature_means[j], features[:, j, :]
        )
        scaler = StandardScaler()
        normalized[:, j, :] = scaler.fit_transform(channel_data.reshape(-1, 1)).reshape(
            n_records, n_timesteps
        )

    # 4. Признаки с выбросами (st_deviation)
    j = 11
    channel_data = np.where(
        np.isnan(features[:, j, :]), feature_means[j], features[:, j, :]
    )
    scaler = RobustScaler()
    normalized[:, j, :] = scaler.fit_transform(channel_data.reshape(-1, 1)).reshape(
        n_records, n_timesteps
    )

    # 5. Амплитудные признаки (qrs_amplitude)
    j = 12
    channel_data = np.where(
        np.isnan(features[:, j, :]), feature_means[j], features[:, j, :]
    )
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized[:, j, :] = scaler.fit_transform(channel_data.reshape(-1, 1)).reshape(
        n_records, n_timesteps
    )

    # 6. Вейвлет-коэффициенты (каналы 13-20)
    for j in [15, 16, 17, 18]:
        channel_data = np.where(
            np.isnan(features[:, j, :]), feature_means[j], features[:, j, :]
        )
        scaler = StandardScaler()
        normalized[:, j, :] = scaler.fit_transform(channel_data.reshape(-1, 1)).reshape(
            n_records, n_timesteps
        )

    # 7. HRV показатели (логарифмическая нормализация)
    for j in [13, 14]:
        # Заменяем NaN и добавляем 1 перед логарифмированием
        channel_data = np.where(
            np.isnan(features[:, j, :]), feature_means[j], features[:, j, :]
        )
        log_data = np.log1p(channel_data)
        scaler = StandardScaler()
        normalized[:, j, :] = scaler.fit_transform(log_data.reshape(-1, 1)).reshape(
            n_records, n_timesteps
        )

    return normalized


# Пример использования:
# features = np.random.rand(900, 23, 5000)
# features[10, 5, 1000] = np.nan  # Добавляем тестовый NaN
# normalized = normalize_ecg_features(features)
# print("Остались ли NaN после обработки:", np.isnan(normalized).any())
