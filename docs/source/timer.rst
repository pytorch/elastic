Expiration Timers
==================

.. automodule:: torchelastic.timer
.. currentmodule:: torchelastic.timer

Client Methods
---------------
.. autofunction:: torchelastic.timer.configure

.. autofunction:: torchelastic.timer.expires

Server/Client Implementations
------------------------------
Below are the timer server and client pairs that are provided by torchelastic.

.. note:: Timer server and clients always have to be implemented and used
          in pairs since there is a messaging protocol between the server
          and client.

.. autoclass:: LocalTimerServer

.. autoclass:: LocalTimerClient

Writing a custom timer server/client
--------------------------------------

To write your own timer server and client extend the
``torchelastic.timer.TimerServer`` for the server and
``torchelastic.timer.TimerClient`` for the client. The
``TimerRequest`` object is used to pass messages between
the server and client.

.. autoclass:: TimerRequest
   :members:

.. autoclass:: TimerServer
   :members:

.. autoclass:: TimerClient
   :members:



