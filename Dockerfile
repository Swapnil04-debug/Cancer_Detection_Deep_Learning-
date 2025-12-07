FROM python:3.10

WORKDIR /code

COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /code/

ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
