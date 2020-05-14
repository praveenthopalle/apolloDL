"""
SessionSecurityMiddleware is the heart of the security that this application
attemps to provide.

To install this middleware, add to your ``settings.MIDDLEWARE_CLASSES``::

    'session_security.middleware.SessionSecurityMiddleware'

Make sure that it is placed **after** authentication middlewares.
"""

from datetime import datetime, timedelta

from django.contrib.auth import logout
from django.core.exceptions import ImproperlyConfigured
from django.urls import reverse, resolve, Resolver404
from django.conf import settings

try:
    from django.utils.deprecation import MiddlewareMixin
except ImportError:  # Django < 1.10
    # Works perfectly for everyone using MIDDLEWARE_CLASSES
    MiddlewareMixin = object

from .utils import get_last_activity, set_last_activity
from apollo.settings import SESSION_SECURITY_EXPIRE_AFTER


class SessionSecurityMiddleware(MiddlewareMixin):
    """
    In charge of maintaining the real 'last activity' time, and log out the
    user if appropriate.
    """

    def __init__(self, get_response):
        self.get_response = get_response
        if not settings.SESSION_EXPIRE_AT_BROWSER_CLOSE:
            raise ImproperlyConfigured(
                'SessionTimeoutMiddleware expects '
                'SESSION_EXPIRE_AT_BROWSER_CLOSE to be True.')

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def is_passive_request(self, request):
        """ Should we skip activity update on this URL/View. """
        if request.path in 'svl':
            return True

        try:
            match = resolve(request.path)
            # TODO: check namespaces too
            if match.url_name in 'svl':
                return True
        except Resolver404:
            pass

        return False

    def get_expire_seconds(self, request):
        """Return time (in seconds) before the user should be logged out."""
        return SESSION_SECURITY_EXPIRE_AFTER

    def process_request(self, request):
        """ Update last activity time or logout. """
        if not request.user.is_authenticated:
            return

        now = datetime.now()
        if '_session_security' not in request.session:
            set_last_activity(request.session, now)
            return

        delta = now - get_last_activity(request.session)
        expire_seconds = self.get_expire_seconds(request)
        if delta >= timedelta(seconds=expire_seconds):
            logout(request)
        elif (request.path == reverse('svl') and
              'idleFor' in request.GET):
            self.update_last_activity(request, now)
        elif not self.is_passive_request(request):
            set_last_activity(request.session, now)

    def update_last_activity(self, request, now):
        """
        If ``request.GET['idleFor']`` is set, check if it refers to a more
        recent activity than ``request.session['_session_security']`` and
        update it in this case.
        """
        last_activity = get_last_activity(request.session)
        server_idle_for = (now - last_activity).seconds

        # Gracefully ignore non-integer values
        try:
            client_idle_for = int(request.GET['idleFor'])
        except ValueError:
            return

        # Disallow negative values, causes problems with delta calculation
        if client_idle_for < 0:
            client_idle_for = 0

        if client_idle_for < server_idle_for:
            # Client has more recent activity than we have in the session
            last_activity = now - timedelta(seconds=client_idle_for)

        # Update the session
        set_last_activity(request.session, last_activity)

    def get_expire_at_browser_close(self):
        """
        Returns ``True`` if the session is set to expire when the browser
        closes, and ``False`` if there's an expiry date. Use
        ``get_expiry_date()`` or ``get_expiry_age()`` to find the actual expiry
        date/age, if there is one.
        """
        if self.get('_session_expiry') is None:
            return settings.SESSION_EXPIRE_AT_BROWSER_CLOSE
        return self.get('_session_expiry') == 0


