# Решение команды Assistant в рамках хакатона по предиктивной аналитике(Sentiement Analysis)
## Задачи
1. Разработайте модель машинного обучения, которая анализирует текстовые отзывы пользователей на продукты или
услуги и классифицирует их как положительные, отрицательные или нейтральные. Учтите, что отзывы могут содержать
сложные языковые конструкции, сарказм и неоднозначные выражения, что усложняет задачу классификации.

2. На основании данных отзывов определите ключевые драйверы роста (что нравится пользователям) и ключевые
барьеры (что не нравится пользователям) развития продукта или услуг.

3. Разработайте рекомендации для маркетингового отдела на основе результатов модели, чтобы они могли
сформировать стратегию развития и позиционирования продукта или услуг.

В рамках задачи 1 мы отобрали в ходе экспериментов архитектуру BERT и взяли за основу веса с https://huggingface.co/blanchefort/rubert-base-cased-sentiment. Также мы решили использовать https://huggingface.co/r1char9/rubert-base-cased-russian-sentiment, бладгодаря чему модель более точно классифицирует неоднозначные выражения

В рамках задачи 2 мы использовали BertTopic и NER. 

Задачу Named Entity Recognition, решили используя BERT, обученный на датасете https://huggingface.co/datasets/RCC-MSU/collection3 с точностью по метрике F1 score = 0.985

В рамках задачи 3 мы решили использовать LLM и в качестве примера использовали LLM от Яндекса, поскольку она показала более хорошие результаты по сравнению с её open-source аналогами. Её можно заменить на любую другую и развернуть в контуре компании. Однако, поскольку в наши задачи не входила разработка LLM, мы решили воспользоваться готовым решением.

В качестве интерфейса, нами было разработано API и tg-bot. Первое позволит быстро и эффективно внедрить в процессы компании, а второе - проверить работоспособность решения, а также для рассмотрения отдельных случаев. Помимо простого текстового запроса вы можете отправить боту таблицу(в формате .csv) и он ее обработает выдаст результат
