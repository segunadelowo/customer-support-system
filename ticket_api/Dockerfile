FROM golang:1.21-alpine

WORKDIR /app

COPY go.mod ./
COPY go.sum ./
RUN go mod download
RUN go mod tidy

COPY . .

RUN go build -o ticket_api

EXPOSE 8080

CMD ["./ticket_api"]