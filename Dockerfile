FROM python:3.10-slim

WORKDIR /app

ADD . /app

RUN python -m pip install --upgrade pip
RUN pip install numpy nltk pandas scikit-learn

EXPOSE 80

CMD ["python", "./NaiveBayesFilter.py"]

