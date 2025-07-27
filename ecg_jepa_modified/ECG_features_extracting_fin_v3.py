# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 10:12:07 2025

@author: padin
"""

import numpy as np
from scipy.signal import find_peaks
import pywt
from tqdm import tqdm



##########################################физические свойства ЭКГ###############################

def detect_r_peaks(ecg_signal, fs=500):
    """Улучшенный детектор R-пиков с адаптивным порогом"""
    # Адаптивный порог для разных участков сигнала
    threshold = np.percentile(ecg_signal, 95) * 0.7
    peaks, _ = find_peaks(ecg_signal, height=threshold, 
                         distance=int(fs*0.6),  # Минимальное расстояние 600 мс
                         prominence=0.5)  # Минимальная высота пика
    return peaks

def compute_heart_rate(r_peaks, fs=500):
    """Вычисляет ЧСС с проверкой валидности RR-интервалов"""
    if len(r_peaks) < 2: 
        return 0
    
    rr_intervals = np.diff(r_peaks) / fs
    # Фильтрация артефактов (интервалы вне 0.4-1.5 сек)
    valid_rr = rr_intervals[(rr_intervals > 0.4) & (rr_intervals < 1.5)]
    
    if len(valid_rr) == 0:
        return 0
    return 60 / np.mean(valid_rr)

def compute_rolling_hr(r_peaks, fs=500, window_sec=10):
    """Скользящее ЧСС с окном в секундах"""
    if len(r_peaks) < 2: 
        return np.zeros(len(r_peaks))
    
    rr_intervals = np.diff(r_peaks) / fs
    window_size = int(window_sec * fs / np.mean(np.diff(r_peaks)))
    
    # Гауссово окно вместо равномерного для сглаживания
    window = np.exp(-np.linspace(-3, 3, window_size)**2)
    window /= window.sum()
    
    rolling_hr = 60 / np.convolve(rr_intervals, window, mode='same')
    return np.pad(rolling_hr, (0, 1), 'edge')  # Сохраняем длину

def compute_hrv_features(r_peaks, fs=500):
    """Улучшенный расчет HRV с фильтрацией артефактов"""
    if len(r_peaks) < 3: 
        return 0.0, 0.0
    
    rr_intervals = np.diff(r_peaks) / fs
    # Фильтрация артефактов
    valid_rr = rr_intervals[(rr_intervals > 0.4) & (rr_intervals < 1.5)]
    
    if len(valid_rr) < 2:
        return 0.0, 0.0
    
    sdnn = np.std(valid_rr) * 1000  # В миллисекундах
    rmssd = np.sqrt(np.mean(np.square(np.diff(valid_rr)))) * 1000
    return sdnn, rmssd

def compute_st_deviation(ecg_signal, r_peaks, fs=500):
    """Точный расчет ST-девиации с базовой линией"""
    if len(r_peaks) < 2:
        return np.zeros(len(ecg_signal))
    
    st_dev = np.zeros(len(ecg_signal))
    for i, peak in enumerate(r_peaks[:-1]):
        # Базовый уровень (PQ-сегмент)
        pq_start = max(0, peak - int(0.2*fs))
        pq_end = peak - int(0.05*fs)
        baseline = np.mean(ecg_signal[pq_start:pq_end])
        
        # ST-сегмент (J-point + 60-80 мс)
        st_start = min(peak + int(0.08*fs), len(ecg_signal)-1)
        st_end = min(st_start + int(0.12*fs), len(ecg_signal)-1)
        
        if st_end <= st_start:
            continue
            
        st_dev[st_start:st_end] = ecg_signal[st_start:st_end] - baseline
    
    return st_dev

def compute_qrs_amplitude(ecg_signal, r_peaks):
    """Нормализованная амплитуда QRS"""
    if len(r_peaks) == 0:
        return np.zeros(len(ecg_signal))
    
    # Медианная амплитуда для устойчивости к выбросам
    median_amp = np.median(ecg_signal[r_peaks])
    return np.full(len(ecg_signal), median_amp)

def compute_swt_features(ecg_signal, wavelet='db4', level=4):
    """Оптимизированное стационарное вейвлет-преобразование с 4 уровнями"""
    max_level = pywt.swt_max_level(len(ecg_signal))
    if level > max_level:
        level = max_level
    
    coeffs = pywt.swt(ecg_signal, wavelet, level=min(level, 4))  # Ограничиваем 4 уровнями
    features = np.zeros((len(ecg_signal), 4))  # 4 детализирующих + 1 аппроксимирующий
    
    # Аппроксимационные коэффициенты (последний уровень)
    features[:, 0] = coeffs[-1][0] if len(coeffs) > 0 else np.zeros_like(ecg_signal)
    
    # Детализирующие коэффициенты (первые 4 уровня)
    for i in range(1, 4):
        if i-1 < len(coeffs):
            features[:, i] = coeffs[i-1][1]
        else:
            features[:, i] = np.zeros_like(ecg_signal)
    
    return features

def extract_ecg_features(ecg_data, fs=500):
    """Оптимизированная функция извлечения признаков с 4 вейвлет-уровнями"""
    num_samples, num_leads = ecg_data.shape
    lead_ii = ecg_data[:, 1]  # II отведение
    
    # Детекция R-пиков
    r_peaks = detect_r_peaks(lead_ii, fs)
    
    # Вычисление признаков
    hr = np.full(num_samples, compute_heart_rate(r_peaks, fs))
    rolling_hr = np.full(num_samples, compute_rolling_hr(r_peaks, fs)[0] if len(r_peaks) > 1 else 0)
    st_dev = compute_st_deviation(lead_ii, r_peaks, fs)
    qrs_amp = compute_qrs_amplitude(lead_ii, r_peaks)
    sdnn, rmssd = compute_hrv_features(r_peaks, fs)
    
    # Вейвлет-признаки (4 уровня)
    wavelet_coeffs = compute_swt_features(lead_ii, level=4)
    
    # Создание финальной матрицы признаков
    feature_matrix = np.column_stack([
        np.zeros(num_samples),  # placeholder для r_peaks
        hr,
        rolling_hr,
        st_dev,
        qrs_amp,
        np.full(num_samples, sdnn),
        np.full(num_samples, rmssd),
        wavelet_coeffs  # автоматически развернется в 5 столбцов
    ])
    
    # Заполнение R-пиков
    if len(r_peaks) > 0:
        feature_matrix[r_peaks, 0] = 1
    
    return feature_matrix

###########################



def augment_ecg_channels(ecg_data, fs=500):
    """
    Основная функция:
    Вход: (10, 8, 2500) - 10 замеров, 8 отведений, 2500 точек
    Выход: (10, 34, 2500) - 8 исходных + 11 признаковых отведений
    """
    num_measurements, num_leads, signal_length = ecg_data.shape
    n_features = 11  # Общее количество признаков
    
    # Создаем расширенный массив (10, 34, 2500)
    augmented_data = np.zeros((num_measurements, num_leads + n_features, signal_length))
    
    for meas_idx in tqdm(range(num_measurements), desc="Обработка замеров"):
        # Копируем исходные 8 отведений
        augmented_data[meas_idx, :num_leads, :] = ecg_data[meas_idx, :, :]
        signals = np.transpose(ecg_data[meas_idx, :, :])
        
        features = extract_ecg_features(signals)
        features = np.transpose(features)
        
        # # Для каждого из 8 исходных отведений извлекаем признаки
        feature_offset = num_leads
        # for lead_idx in range(num_leads):
        #     # Извлекаем признаки
        #     features = extract_lead_features(ecg_data[meas_idx, lead_idx, :], fs)
            
        #     # Создаем новые "отведения" с дублированием признаков по времени
        for feat_idx in range(len(features)):
                augmented_data[meas_idx, feature_offset + feat_idx, :] = np.full(signal_length, features[feat_idx])
            
            # feature_offset += n_features
    
    return augmented_data