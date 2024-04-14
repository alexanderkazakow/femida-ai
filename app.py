import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

st.set_page_config(layout="wide", page_title="Femida AI", page_icon="⚖️")
metadata = {
    "title": "Femida AI",
    "description": "Femida AI - это мощное веб-приложение, созданное для автоматизации процесса загрузки и классификации документов. Независимо от того, нужны вам юридические документы для работы или для личных целей, наше приложение обеспечит быструю и эффективную обработку текстовых файлов",
    "author": ["КибергенИИ"],
    "email": "gallininadia@gmail.com",
    "developers": ["Галлини Надежда", "Казаков Александр", "Деревянко Александр", "Зеленский Сергей", "Мишина Лилия"],
}

def main():
    import streamlit as st
    import pandas as pd
    import magic
    import csv
    import sys
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    import re
    import io
    import pyth.plugins.rtf15.reader as rtf_reader
    import io
    with st.sidebar: 
         st.image("media/logo.png")
         choice = st.radio("Навигация", ["Загрузка файлов", "Классификация", "Обучить модель", "Анализ"])
         st.info("Это проектное приложение классифицирует файлы.")
         st.title("КибергенИИ")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        pass
    with col2:
        pass
    with col3:
        pass
    with col4:
        pass
    with col5:
        pass
    with col6:
         st.image("media/femida.png", width=200)

    if choice == "Загрузка файлов":
        import pandas as pd
        import streamlit as st
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        import joblib
        import magic
        import sys
        import os

        # Функция для загрузки обученной модели
        def load_trained_model():
            try:
                model = joblib.load('trained_model.pkl')
                return model
            except FileNotFoundError:
                st.error("Файл trained_model.pkl не найден!")
                return None

        # Функция для загрузки TF-IDF векторизатора
        def load_vectorizer():
            try:
                vectorizer = joblib.load('vectorizer.pkl')
                return vectorizer
            except FileNotFoundError:
                st.error("Файл vectorizer.pkl не найден!")
                return None

        # Функция для классификации документа с использованием обученной модели
        def classify_document(file_content, model, vectorizer):
            if model:
                # Преобразование содержимого файла с использованием TF-IDF векторизатора
                file_content_vectorized = vectorizer.transform([file_content])
                
                # Прогнозирование с помощью модели
                document_type = model.predict(file_content_vectorized)[0]
                return document_type
            else:
                return "Unknown"

        # Функция для определения класса документа по его имени
        # def get_document_class(filename):
        #     if 'dogovor' in filename.lower():
        #                 return 'contract'
        #     elif 'zakon' in filename.lower():
        #         return 'alf'
        #     elif 'ustav' in filename.lower():
        #         return 'statute'
        #     elif 'contracts' in filename.lower():
        #             return 'contracts'
        #     elif 'rasporiazhenie' in filename.lower():
        #             return 'resolution'
        #     elif 'prilozhenie' in filename.lower():
        #             return 'bill'
        #     elif 'prilozhenie' in filename.lower():
        #             return 'bill'
        #     else:
        #         return 'Unknown'
        # Define the dataset
        document_classes = {
            'доверенность': 'proxy',
            'dogovor': 'contract',
            'акт': 'act',
            'zayavleniye': 'application',
            'prikaz': 'order',
            'schet': 'invoice',
            'prilozhenie': 'bill',
            'agreement': 'arrangement',
            'dogovor oferty': 'contract_offer',
            'ustav': 'statute',
            'resheniye': 'determination',
            'zakon': 'law',
            'rasporiazhenie': 'resolution',
            'contracts': 'contracts'
        }

        # Function for determining the document class
        def get_document_class(filename):
            lowercase_filename = filename.lower()
            for key, value in document_classes.items():
                if key in lowercase_filename:
                    return value
            return 'Unknown'

        # Test the function
        print(get_document_class('доверенность'))  # Output: proxy
        print(get_document_class('dogovor'))        # Output: contract
        print(get_document_class('решение'))        # Output: determination
        print(get_document_class('unknown'))        # Output: Unknown


        st.title("Загрузка и классификация документов")

        uploaded_files = st.file_uploader("Выберите документы", accept_multiple_files=True)

        if uploaded_files:
            # Загрузка обученной модели и TF-IDF векторизатора
            model = load_trained_model()
            vectorizer = load_vectorizer()
            
            if model and vectorizer:
                uploaded_info = []
                
                for file in uploaded_files:
                    filename = file.name
                    
                    # Проверка имени файла
                    if not filename:
                        st.warning("Файл без имени не может быть обработан. Пропуск.")
                        continue
                        
                    file_content = file.getvalue().decode('utf-8')
                    
                    # Определение класса документа по имени файла
                    document_class = get_document_class(filename)
                    
                    # Если класс документа неизвестен, пропускаем файл
                    if document_class == 'Unknown':
                        continue
                        
                    # Определение типа файла и его размер
                    mime = magic.Magic(mime=True)
                    file.seek(0)
                    file_format = mime.from_buffer(file.read(1024))
                    file_size = sys.getsizeof(file_content)
                    
                    # Если файл формата text/html, пропускаем его
                    if file_format == 'text/html':
                        continue
                        
                    # Классификация документа
                    document_type = classify_document(file_content, model, vectorizer)
                    
                    # Сбор информации о загруженных документах
                    uploaded_info.append({
                        'Название': filename,
                        'Формат': file_format,
                        'Размер': file_size,
                        'Классификация': document_class,
                    })

                df = pd.DataFrame(uploaded_info)
                st.write("### Информация о загруженных документах")
                st.write(df)
#Классификация
        import pandas as pd

    def append_to_csv(new_data):
        try:
            # Попытка чтения существующего CSV файла
            existing_data = pd.read_csv('data.csv')
            # Добавление новых данных к существующим данным
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        except (FileNotFoundError, pd.errors.ParserError):
            # Если файл не существует или произошла ошибка парсинга, создаем новый DataFrame
            updated_data = new_data

        # Запись обновленных данных в CSV файл
        updated_data.to_csv('data.csv', index=False)

    if choice == "Классификация":  
        st.title('Добавление класса и текста в CSV файл')
        if st.checkbox('Показать содержимое файла data.csv'):
            try:
                df_content = pd.read_csv('data.csv')
                st.write(df_content)
            except FileNotFoundError:
                st.error("Файл data.csv не найден!")
        
        # Создание текстовых полей для ввода класса и текста
        class_name = st.text_input('Введите класс:', key='class_input')
        text = st.text_area('Введите текст:', key='text_input')

        # Кнопка для добавления класса и текста в CSV файл
        if st.button('Добавить в CSV'):
            # Создание DataFrame для новой записи
            new_data = pd.DataFrame({'class': [class_name], 'text': [text]})
            # Запись в CSV файл
            append_to_csv(new_data)
            st.success('Данные успешно добавлены в CSV файл.')
   
     #Обучение модели           
    import pandas as pd
    import streamlit as st
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import LinearSVC
    import joblib

    # Функция для обучения модели
    def train_model(data, test_size):
        # Предобработка данных
        X = data['text']
        y = data['class']

        # Векторизация текста
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X)

        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Обучение модели
        model = LinearSVC()
        model.fit(X_train, y_train)

        # Оценка модели
        accuracy = model.score(X_test, y_test)

        return model, accuracy

    # Функция для сохранения модели в файл
    def save_model(model):
        # Сохранение модели в файл trained_model.pkl
        joblib.dump(model, 'trained_model.pkl')

        # Сообщение о успешном сохранении
        st.write("Модель успешно сохранена")

    if choice == "Обучить модель":
            
            # текст описания
            description_text = """  
            * В этой части кода данные для обучения модели читаются из файла 'data.csv'.
            * Датасет содержит столбец 'text' с текстом документов и столбец 'class' с метками классов.
            * Текст векторизуется с использованием TF-IDF.
            * Данные разделяются на обучающую и тестовую выборки с помощью функции train_test_split.
            * Обучается модель Linear Support Vector Classification (LinearSVC) из библиотеки scikit-learn.
            * Оценка модели производится с помощью функции score, которая вычисляет точность предсказаний на тестовой выборке.
            """

            # Вывод текста на страницу
            st.markdown(description_text)

            # Загрузка данных
            uploaded_data = st.file_uploader("Выберите файл с данными для обучения модели", type=["csv", "xlsx"])

            # Выбор параметров модели
            test_size = st.slider("Выберите размер тестовой выборки", min_value=0.1, max_value=0.5, step=0.1, value=0.2)

            if uploaded_data is not None:
                # Чтение данных
                if uploaded_data.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                    df = pd.read_excel(uploaded_data)
                else:
                    df = pd.read_csv(uploaded_data)
                
                # Создание и обучение TF-IDF векторизатора
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(df['text'])  # Предполагается, что тексты находятся в столбце 'text'

                # Сохранение векторизатора в файл
                joblib.dump(vectorizer, 'vectorizer.pkl')

                # Кнопка для обучения модели
                if st.button("Обучить модель"):
                    # Обучение модели
                    model, accuracy = train_model(df, test_size)

                    # Сохранение обученной модели
                    save_model(model)

                    st.write(f"Точность модели: {accuracy}")




    if choice == "Анализ":
        from pandas_profiling import ProfileReport
        from streamlit_pandas_profiling import st_profile_report
        st.title("Загрузка данных")
        file = st.file_uploader("Загрузите свой набор данных")
        if file: 
            df = pd.read_csv(file, sep=',', index_col=None)
            df.to_csv('dataset.csv', index=None)
            
            # Отображение таблицы данных с улучшенным стилем
            st.subheader("Пример данных:")
            st.dataframe(df.style.highlight_max(axis=0))  # Пример стилизации таблицы с выделением максимальных значений
            
            st.title("Автоматизированный исследовательский анализ данных")
            profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
            
            # Отображение отчета о профиле данных с улучшенным стилем
            st.header("Отчет о профиле данных:")
            st_profile_report(profile)

if __name__ == '__main__':
    main()
