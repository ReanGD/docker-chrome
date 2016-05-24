FROM reangd/base-x11:latest

ADD https://dl.google.com/linux/linux_signing_key.pub /tmp/
COPY run_chrome /bin/

RUN chmod +x /bin/run_chrome && \
	apt-get install -y --no-install-recommends dbus-x11 libexif12 && \
	apt-key add /tmp/linux_signing_key.pub && \
	echo 'deb http://dl.google.com/linux/chrome/deb/ stable main' >> /etc/apt/sources.list && \
	apt-get update && \
	apt-get install -y google-chrome-stable && \
	apt-get clean -y

CMD /bin/run_chrome
