from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()


class FileUploadForm(forms.Form):
    file_source = forms.FileField()


class LoginForm(forms.ModelForm):
    print('AuthenticationForm', AuthenticationForm)
    username = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': "input"}),
    )
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': "input"}))


