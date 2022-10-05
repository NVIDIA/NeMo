FROM python:3.9

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8001

COPY ./spacy_app.py .

ENTRYPOINT ["streamlit", "run", "--server.headless", "true", "--server.port", "8001"]

CMD ["spacy_app.py"]
