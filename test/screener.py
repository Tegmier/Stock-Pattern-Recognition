import blpapi
from connection import test_bloomberg_connection, create_session
import datetime

ticker = "AAPL US Equity"
# start_date = "20240101"
# end_date = "20240501"

test_bloomberg_connection()
session = create_session()
session.start()

session.openService("//blp/screener")
service = session.getService("//blp/screener")
status = service.isValid()

request = service.createRequest("BeqsRequest")
request.set("screenName", "Japan Semiconductor Equities list")
request.set("screenType", "PRIVATE")


session.sendRequest(request)

while True:
    ev = session.nextEvent()
    for msg in ev:
        print(msg)

    if ev.eventType() == blpapi.Event.RESPONSE:
            break

session.stop()
