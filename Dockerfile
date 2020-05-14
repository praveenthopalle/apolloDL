FROM python:3.7

MAINTAINER praveenthopalle

EXPOSE 8000

ADD . /apollo

WORKDIR /apollo

RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

RUN pip install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN python manage.py makemigrations

RUN python manage.py migrate

CMD [ "python", "manage.py", "runserver", "0.0.0.0:8000" ]