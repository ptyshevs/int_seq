# Описание

Следуя [этой](https://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/) статье,
я попробовал повторить схожую архитектуру. Идея такова: вместо того, чтобы переводить последовательность в числа,
мы используем её как последовательность символов. RNN генерирует выходную последовательность, которая является следующим
числом в последовательности.

# Что пошло не так?

Этот способ неплохо себя показал для предсказания следующего члена в арифметической\геометрической прогрессии, но наше задание оказалось гораздо сложнее (ещё бы).
    Любопытным исходом оказалось то, что модель научилась довольно точно определять **количество цифр**, но не значения чисел. То есть, если следующее значение имеет 10 цифр, а максимальная длина установлена в 40, модель почти всегда успешно
предсказывает 30 символов "отступа", заполняя оставшиеся 10 цифр неправильными значениями. Ясное дело, это не то, для чего модель была задумана.

# Как исправить?

Есть несколько путей решения:

1) Использовать `batch_size=1` и обучать на последовательностях различной длины (возможно, слишком долго).
2) Вернуться к постановке проблемы как задачи регрессии -- последний слой состоит из одного нейрона, loss - 'mae' или что-то подобное.
    