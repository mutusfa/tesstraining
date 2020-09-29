FROM python:3.8-alpine

RUN apk add --no-cache tesseract-ocr

ENTRYPOINT [tesseract]
