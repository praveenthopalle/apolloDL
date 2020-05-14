"""
WSGI config for apollo project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/
"""

import os
import django.core.handlers.wsgi

os.environ["DJANGO_SETTINGS_MODULE"] = "apollo.settings"

application = django.core.handlers.wsgi.WSGIHandler()


