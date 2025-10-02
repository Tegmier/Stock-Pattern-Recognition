import blpapi
from blpapi import SessionOptions, Session


options = SessionOptions()
options.setServerHost('localhost')  # 默认主机
options.setServerPort(8194)         # 默认端口

session = Session(options)
session.start()

session.openService("//blp/refdata")
service = session.getService("//blp/refdata")
status = service.isValid()

print(status)
