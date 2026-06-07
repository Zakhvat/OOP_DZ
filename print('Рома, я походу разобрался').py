"""
Библиотека градиентного спуска с метрикой MAPE
Студенты: [Хватов Захар(остальная команда допишите каждый себя сам)]
Группа: [Крутые парни]
Вариант: 1 (MAPE)
"""

import numpy as np
import matplotlib.pyplot as plt

class ГрадиентныйСпуск:
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None  # веса модели (коэффициенты)
        self.bias = None     # смещение (intercept)
        self.loss_history = []  # история изменения функции потерь для визуализации
        
    def mape_loss(self, y_true, y_pred):
        """
        Вычисление MAPE (Mean Absolute Percentage Error)
        
        Формула: MAPE = (1/n) * sum(|(y_true - y_pred) / y_true|) * 100%
        
        Параметры:
        y_true (numpy array): истинные значения
        y_pred (numpy array): предсказанные значения
        
        Возвращает:
        float: значение MAPE в процентах
        """
        # Добавляем маленькое число, чтобы избежать деления на ноль
        epsilon = 1e-8
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        percentage_error = np.abs((y_true - y_pred) / (y_true + epsilon))
        
        percentage_error = np.nan_to_num(percentage_error, nan=0.0, posinf=0.0, neginf=0.0)
        
        mape = np.mean(percentage_error) * 100
        
        return mape
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def compute_gradients(self, X, y, y_pred):
        n_samples = len(y)
        
        # Градиенты для MSE: d/dw = (2/n) * X.T.dot(y_pred - y)
        #                                 d/db = (2/n) * sum(y_pred - y)
        dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (2 / n_samples) * np.sum(y_pred - y)
        
        return dw, db
    
    def fit(self, X, y, verbose=True):
        # Преобразуем данные в numpy массивы, если это списки
        X = np.array(X)
        y = np.array(y)
        
        # Получаем размерность данных
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Очищаем историю потерь
        self.loss_history = []
        
        print("=" * 60)
        print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
        print("=" * 60)
        print(f"Количество образцов: {n_samples}")
        print(f"Количество признаков: {n_features}")
        print(f"Скорость обучения: {self.learning_rate}")
        print(f"Максимальное число итераций: {self.max_iterations}")
        print(f"Критерий остановки: {self.tolerance}")
        print("=" * 60)
        
        for iteration in range(self.max_iterations):
            # Предсказываем значения
            y_pred = self.predict(X)
            
            # Вычисляем MAPE для текущей модели (для оценки)
            current_mape = self.mape_loss(y, y_pred)
            self.loss_history.append(current_mape)
            
            # Вычисляем градиенты
            dw, db = self.compute_gradients(X, y, y_pred)
            
            # Обновляем веса и смещение
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Выводим информацию о процессе обучения
            if verbose and (iteration % 100 == 0 or iteration == self.max_iterations - 1):
                print(f"Итерация {iteration:4d}: MAPE = {current_mape:.4f}%")
            
            # Проверяем критерий остановки
            if iteration > 0:
                loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
                if loss_change < self.tolerance:
                    print(f"\nДосрочная остановка на итерации {iteration}")
                    print(f"Изменение функции потерь: {loss_change:.2e} < {self.tolerance}")
                    break
        
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)
        print(f"Финальный MAPE: {current_mape:.4f}%")
        print(f"Веса модели: {self.weights}")
        print(f"Смещение (bias): {self.bias:.4f}")
        print("=" * 60)
        
        return self
    
    def plot_loss_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, 'b-', linewidth=2)
        plt.title('Динамика изменения MAPE в процессе обучения', fontsize=14)
        plt.xlabel('Итерация', fontsize=12)
        plt.ylabel('MAPE (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mape = self.mape_loss(y_test, y_pred)
        print(f"\nОценка на тестовых данных: MAPE = {mape:.4f}%")
        return mape
    
    def predict_with_confidence(self, X):
        predictions = self.predict(X)
        print("\nРезультаты предсказания:")
        print("-" * 50)
        for i, pred in enumerate(predictions):
            print(f"Объект {i+1}: предсказанное значение = {pred:.4f}")
        return predictions

def демонстрация_работы_библиотеки():
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ БИБЛИОТЕКИ")
    print("=" * 60)

    print("\n1. Генерация тестовых данных...")
    np.random.seed(42)  # Для воспроизводимости результатов
    n_samples = 100
    X = np.random.randn(n_samples, 1)  # Один признак
    true_weights = 3
    true_bias = 2
    y = true_weights * X.flatten() + true_bias + np.random.randn(n_samples) * 0.5  # Добавляем шум
    
    # Разделяем данные на обучающую и тестовую выборки (80% / 20%)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Обучающая выборка: {len(X_train)} объектов")
    print(f"Тестовая выборка: {len(X_test)} объектов")
    print(f"Истинная зависимость: y = {true_weights} * x + {true_bias}")
    
    # 2. Создаём и обучаем модель
    print("\n2. Инициализация модели градиентного спуска...")
    model = ГрадиентныйСпуск(
        learning_rate=0.1,
        max_iterations=1000,
        tolerance=1e-6
    )
    
    # 3. Обучаем модель
    print("\n3. Запуск обучения...")
    model.fit(X_train, y_train, verbose=True)
    
    # 4. Визуализируем процесс обучения
    print("\n4. Визуализация процесса обучения...")
    model.plot_loss_history()
    
    # 5. Оцениваем модель на тестовых данных
    print("\n5. Оценка качества модели...")
    test_mape = model.evaluate(X_test, y_test)
    
    # 6. Делаем предсказания
    print("\n6. Предсказания модели...")
    predictions = model.predict_with_confidence(X_test[:5])  # Первые 5 тестовых объектов
    
    # 7. Сравниваем предсказания с истинными значениями
    print("\n7. Сравнение предсказаний с реальными значениями:")
    print("-" * 50)
    for i in range(5):
        print(f"Объект {i+1}: Истинное = {y_test[i]:.4f}, Предсказанное = {predictions[i]:.4f}, "
              f"Ошибка = {abs(y_test[i] - predictions[i]):.4f}")
    
    # Выводим итоговый отчёт
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    print(f"✓ Модель успешно обучена")
    print(f"✓ Используемая метрика: MAPE (Mean Absolute Percentage Error)")
    print(f"✓ Точность модели на тестовых данных: {100 - test_mape:.2f}%")
    print(f"✓ Средняя ошибка предсказания: {test_mape:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    демонстрация_работы_библиотеки()
    
    # Дополнительный пример с ручным вводом данных
    print("\n\n" + "=" * 60)
    print("ПРОСТОЙ ПРИМЕР РУЧНОГО ИСПОЛЬЗОВАНИЯ")
    print("=" * 60)
    
    # Данные: площадь квартиры (м²) -> цена (тыс. $)
    X_manual = [[30], [40], [50], [60], [70], [80]]
    y_manual = [45, 60, 75, 90, 105, 120]
    
    # Создаём и обучаем модель
    model_manual = ГрадиентныйСпуск(learning_rate=0.01, max_iterations=500)
    model_manual.fit(X_manual, y_manual, verbose=False)
    
    # Делаем предсказание для квартиры 55 м²
    new_apartment = [[55]]
    predicted_price = model_manual.predict(new_apartment)
    print(f"\nПредсказание для квартиры 55 м²: {predicted_price[0]:.2f} тыс. $")
    
    # Оцениваем точность модели на исходных данных
    mape_train = model_manual.mape_loss(y_manual, model_manual.predict(X_manual))
    print(f"Точность модели на обучающих данных: {100 - mape_train:.2f}%")
    print(f"Средняя ошибка: {mape_train:.2f}%")
print('Бабулька 3')
<<<<<<< HEAD
print('ну допустим дедушка')
#ну пожалуйста всё заработай....
#я моляю....
=======
print('вроде редачится')


# чтоб всё зароботало, аминь
>>>>>>> ba0d42daf125f631892e80c93ce87ef6c5b84565











#Есть изменеия?