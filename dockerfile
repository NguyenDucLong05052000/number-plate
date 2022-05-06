FROM python:3
WORKDIR /home/nguyen/number_plate
COPY . .
EXPOSE 8885
RUN pip install flask
RUN pip install Werkzeug
RUN pip install flask_cors
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install tensorflow
RUN pip install torch
RUN pip install mysql-connector-python
CMD ["python","seasion.py"]