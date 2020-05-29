import string
import sklearn
from config import BOT_CONFIG, BOT_TOKEN, PROXY
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.metrics.distance import edit_distance
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import random
import re
import string
import collections
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer

CLASSIFIER_THRESHOLD = 0.2
GENERATIVE_THRESHOLD = 0.6

with open('dialogues.txt', encoding='utf-8') as f:
    data = f.read()

dialogues = []

for dialogue in data.split('\n\n'):
    replicas = []
    for replica in dialogue.split('\n')[:2]:
        replica = replica[2:].lower()
        replicas.append(replica)

    if len(replicas) == 2 and 5 < len(replicas[0]) < 25 and 5 < len(replicas[1]) < 25:
        dialogues.append(replicas)

GENERATIVE_DIALOGUES = dialogues[:50000]

X_text = []  # тексты
y = []  # и соответствующие им классы

for intent, value in BOT_CONFIG['intents'].items():
    for example in value['examples']:
        X_text.append(example)
        y.append(intent)

# Заменяем спецсимволы на пробел
def subtract_symbols(text):
    chars = re.escape(string.punctuation)
    return re.sub(r'[' + chars + ']', ' ', text)


X_text = list(map(subtract_symbols, X_text))

# Создаём список всех слов
words_list = map(lambda v: v.split(' '), X_text)
flatten_list = lambda l: [item for sublist in words_list for item in sublist]

# Выбираем 5 самых популярных слов в стоплист
counter = collections.Counter(flatten_list(words_list))
stop_words = set(counter.most_common()[:5])

VECTORIZER = CountVectorizer(lowercase=True, ngram_range=(3, 3), analyzer='char', binary=True, stop_words=stop_words)

X = VECTORIZER.fit_transform(X_text)
CLF = LogisticRegression()
CLF.fit(X, y)


def get_intent(text):
    probas = CLF.predict_proba(VECTORIZER.transform([text]))
    max_proba = max(probas[0])
    if float(max_proba) >= float(CLASSIFIER_THRESHOLD):
        index = list(probas[0]).index(max_proba)
        print(repr({max_proba} - {CLF.classes_[index]}))
        return CLF.classes_[index]


def get_answer_by_generative_model(text):
    text = text.lower()
    print(text)
    for question, answer in GENERATIVE_DIALOGUES:
        if abs(len(text) - len(question)) / len(question) < 1 - GENERATIVE_THRESHOLD:
            dist = edit_distance(text, question)
            l = len(question)
            similarity = 1 - dist / l
            print(similarity)
            if similarity > GENERATIVE_THRESHOLD:
                return answer


def get_response_by_intent(intent):
    responses = BOT_CONFIG['intents'][intent]['responses']
    return random.choice(responses)


def get_failure_phrase():
    print('fail')
    phrases = BOT_CONFIG['failure_phrases']
    return random.choice(phrases)


stats = {
    'requests': 0,
    'byscript': 0,
    'bygenerative': 0,
    'stub': 0
}


def generate_answer(text):
    stats['requests'] += 1

    # NLU
    intent = get_intent(text)

    # Make answer

    # by script
    if intent:
        stats['byscript'] += 1
        response = get_response_by_intent(intent)
        return response

    # use generative model
    answer = get_answer_by_generative_model(text)
    if answer:
        stats['bygenerative'] += 1
        return answer

    # use stub
    stats['stub'] += 1
    failure_phrase = get_failure_phrase()
    return failure_phrase


def start(update, context):
    """Команда при старте бота"""
    update.message.reply_text('Можешь написать мне что-нибудь!')


def text(update, context):
    answer = generate_answer(update.message.text)
    print(update.message.text, '->', answer)
    print(stats)
    print()
    update.message.reply_text(answer)


def error(update, context):
    """Выводим ошибку"""
    print(context.error)
    update.message.reply_text('Что-то пошло не так')


def main():
    """Стартуем бота"""
    updater = Updater(BOT_TOKEN,
                      request_kwargs=PROXY,
                      use_context=True)

    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text, text))
    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()


print(generate_answer('Как играть?'))
