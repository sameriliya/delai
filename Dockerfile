FROM docker/getting-started:latest
COPY delai .
COPY dependencies .
RUN install dependencies
CMD launch API web server
