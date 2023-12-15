from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class MLBot:
    def __init__(self):
        # Небольшой набор данных для обучения
        training_data = [
            ("привет", "приветствие"),
            ("как дела", "приветствие"),
            ("что делаешь", "приветствие"),
            ("как тебя зовут", "приветствие"),
            ("какая погода", "погода"),
            ("расскажи шутку", "развлечение"),
            ("какого цвета небо", "небо"),
            # Добавьте свои собственные данные для обучения
        ]

        # Разделяем данные на текст и метки
        texts, labels = zip(*training_data)

        # Создаем модель с использованием метода опорных векторов (SVM)
        self.model = make_pipeline(
            TfidfVectorizer(),
            StandardScaler(with_mean=False),
            SVC(kernel='linear')
        )

        # Обучаем модель на данных
        self.model.fit(texts, labels)

    def get_response(self, user_input):
        # Предсказываем метку для ввода пользователя
        predicted_label = self.model.predict([user_input])[0]

        # Возвращаем соответствующий ответ
        if predicted_label == "приветствие":
            return "Привет! Как я могу вам помочь?"
        elif predicted_label == "погода":
            return "Извините, я не знаю. Я просто бот."
        elif predicted_label == "развлечение":
            return "Почему не стоит доверять атомам? Потому что они состоят из мелочей!"
        elif predicted_label == "небо":
            return "Голубое!"


# Создаем экземпляр бота
bot = MLBot()

# Пример использования
while True:
    user_input = input("Вы: ")

    if user_input.lower() == 'выход':
        break

    response = bot.get_response(user_input)
    print("Бот:", response)