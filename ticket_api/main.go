package main

import (
    "log"
    "net/http"
    "os"
    "encoding/json"

    "github.com/gin-gonic/gin"
    "github.com/streadway/amqp"
)

type TicketRequest struct {
    //Title       string `json:"title" binding:"required"`
    Description string `json:"description" binding:"required"`
    User   string `json:"user" binding:"required"`
}

func main() {
    // Get RabbitMQ URL from environment
    rabbitMQURL := os.Getenv("RABBITMQ_URL")
    if rabbitMQURL == "" {
        rabbitMQURL = "amqp://guest:guest@localhost:5672/"
    }

    // Connect to RabbitMQ
    conn, err := amqp.Dial(rabbitMQURL)
    if err != nil {
        log.Fatalf("Failed to connect to RabbitMQ: %v", err)
    }
    defer conn.Close()

    ch, err := conn.Channel()
    if err != nil {
        log.Fatalf("Failed to open a channel: %v", err)
    }
    defer ch.Close()

    // Declare queue
    q, err := ch.QueueDeclare(
        "ticket_queue", // name
        true,          // durable
        false,         // delete when unused
        false,         // exclusive
        false,         // no-wait
        nil,           // arguments
    )
    if err != nil {
        log.Fatalf("Failed to declare a queue: %v", err)
    }

    // Set up Gin router
    router := gin.Default()

    router.POST("/tickets", func(c *gin.Context) {
        var ticket TicketRequest
        if err := c.ShouldBindJSON(&ticket); err != nil {
            c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
            return
        }

        // Convert ticket to JSON string
        body, err := json.Marshal(ticket)
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to marshal ticket"})
            return
        }

        // Publish message to RabbitMQ
        err = ch.Publish(
            "",     // exchange
            q.Name, // routing key
            false,  // mandatory
            false,  // immediate
            amqp.Publishing{
                ContentType: "application/json",
                Body:        body,
            })
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to publish message"})
            return
        }

        c.JSON(http.StatusCreated, gin.H{
            "message": "Ticket request queued successfully",
            "ticket":  ticket,
        })
    })

    // Start server
    log.Fatal(router.Run(":8080"))
}