import json
import pandas as pd
import pika
import time
from rag_system import RAGSystem
import requests

class TicketProcessor:
    def __init__(self):
        # Connection parameters
        self.credentials = pika.PlainCredentials('guest', 'guest')
        self.parameters = pika.ConnectionParameters(
            host='localhost',
            port=5672,
            credentials=self.credentials
        )
        self.intent_to_response = {}
        self.intent_to_category = {}
        self.queue_names = ['ORDER', 'SHIPPING', 'CANCEL', 'INVOICE', 'PAYMENT', 'REFUND', 'FEEDBACK', 'CONTACT', 'ACCOUNT', 'DELIVERY', 'SUBSCRIPTION', 'REVIEW', 'GENERAL','INTERVENTION']
        
        self.automatable_intents = [
            'check_payment_methods', 
            'get_invoice', 
            'delivery_options', 
            'track_order'
        ]
        self.rag_system = RAGSystem()


    def setup_connection(self):
        # Establish connection and channel
        connection = pika.BlockingConnection(self.parameters)
        channel = connection.channel()
        return connection, channel

    def setup_queues(self, channel):
        # Establish connection and channel
        #channel = connection.channel()
        # Declare queue (creates if doesn't exist)
        for queue_name in self.queue_names:
            channel.queue_declare(queue=queue_name, durable=True)
        

    def load_data(self, file_path):
        """
        Load and preprocess the Bitext dataset.
        
        Args:
            file_path (str): Path to the CSV dataset.
        """
        data = pd.read_csv(file_path)
        self.data = data[['category', 'intent', 'response']]
        # Map intents to their first response for automation
        self.intent_to_response = self.data.groupby('intent')['response'].first().to_dict()
        # Map intents to their categories for routing
        self.intent_to_category = self.data.groupby('intent')['category'].first().to_dict()

    def classify_ticket(self, ticket: str) -> dict:
        """
        Calls a FastAPI endpoint with ticket details and returns the prediction response.
        
        Args:
            endpoint_url (str): The URL of the FastAPI endpoint
            
        Returns:
            dict: Response containing status, predicted_label, prediction_confidence, and sentiment_score
            
        Raises:
            requests.RequestException: If the API call fails
        """
        # Request payload
        request_data = {
            "ticket": ticket
        }
        
        try:
            # Make POST request to the endpoint
            endpoint_url = "http://localhost:8000/predict"
            response = requests.post(
                url=endpoint_url,
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            print(response.json())
            return response.json()
        
        except requests.RequestException as e:
            print(f"Error calling endpoint: {str(e)}")
            return {
                "status": "error",
                "error_message": str(e)
            }

    def get_automated_response(self, intent):
        """
        Retrieve an automated response for a given intent.
        
        Args:
            intent (str): The classified intent.
        
        Returns:
            str: The automated response or a default message.
        """
        return self.intent_to_response.get(intent, "No response available.")

    def decide_action(self, ticket, user):
        """
        Decide the action (automate or route) and priority based on classification and sentiment.
        
        Args:
            ticket (str): The ticket text.
        
        Returns:
            tuple: (action string, priority string)
        """
        # Classify the ticket
        response = self.classify_ticket(ticket)
        intent = response['predicted_label']
        sentiment = response['sentiment_score']
        prob = response['prediction_confidence']

        # Determine routing based on category
        queue_category = self.intent_to_category.get(intent, 'GENERAL')
        
        # Decision logic
        if intent in self.automatable_intents and prob > 0.9:
            response = self.get_automated_response(intent)
            action = f"Automated response: {response}"
            ticket_msg = {"description":ticket, "user":user,"intent": intent, "response": action}
            queue_category = "FEEDBACK"
        else:
            ticket_msg = {"description":ticket, "user":user,"intent": intent}
        
        # Prioritize based on sentiment
        priority = "High" if sentiment < -0.05 else "Normal"
        
        return queue_category, ticket_msg, priority
    
    def consume_messages(self):
        def callback(ch, method, properties, body):

            ticket = json.loads(body.decode())

            queue_category, ticket_msg, priority = self.decide_action(ticket['description'], ticket['user'])

            if queue_category == "SUBSCRIPTION":
                agent_recommendation = self.rag_system.process_ticket(ticket['description'])
                ticket_msg['response'] = agent_recommendation
                self.produce_messages(ticket_msg, "REVIEW")
            elif priority == "High" and queue_category != "SUBSCRIPTION":
                self.produce_messages(ticket_msg, "INTERVENTION")
            else:
                self.produce_messages(ticket_msg, queue_category)


            ch.basic_ack(delivery_tag=method.delivery_tag)

        try:
            connection, channel = self.setup_connection()
            
            # Configure consumer
            channel.basic_qos(prefetch_count=1)  # Process one message at a time
            channel.basic_consume(
                queue='ticket_queue',
                on_message_callback=callback
            )
            
            print("Waiting for messages. Press CTRL+C to exit.")
            channel.start_consuming()

        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error consuming messages: {e}")
        finally:
            channel.close()
            connection.close()

    def produce_messages(self, message, queue_category):
        try:
            connection, channel = self.setup_connection()

            # Send messages
            channel.basic_publish(
                exchange='',
                routing_key=queue_category,
                body=json.dumps(message).encode(),
                properties=pika.BasicProperties(
                    delivery_mode=2  # make message persistent
                )
            )
            print(f"Sent: {message}")
            time.sleep(1)  # Small delay between messages

        except Exception as e:
            print(f"Error producing messages: {e}")
        finally:
            channel.close()
            connection.close()



if __name__ == "__main__":

    ticket_processor = TicketProcessor()
    ticket_processor.rag_system.setup()

    # Load data
    ticket_processor.load_data("data/bitext.csv")

    # Setup queues
    connection, channel = ticket_processor.setup_connection()
    ticket_processor.setup_queues(channel)

    # Run consumer
    print("\nStarting consumer...")
    ticket_processor.consume_messages()

    
    # Give some time between producer and consumer
    time.sleep(2)
    
