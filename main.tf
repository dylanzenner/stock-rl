terraform {

  cloud {
    organization = "cpe-800"

    workspaces {
      name = "cpe-800-project"
    }
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
          }
  }
}


# Configure the AWS Provider
provider "aws" {
  region = "us-east-1"

}

# Create a VPC
resource "aws_vpc" "stock-rl-vpc" {
  cidr_block = "10.0.0.0/16"
}

# Create an Internet Gateway
resource "aws_internet_gateway" "stock-rl-igw" {
  vpc_id = aws_vpc.stock-rl-vpc.id

  tags = {
    Name = "stock-rl-igw"
  }
}

# # Create an Internet Gateway Attachment
# resource "aws_internet_gateway_attachment" "stock-rl-igw-attachment" {
#   vpc_id              = aws_vpc.stock-rl-vpc.id
#   internet_gateway_id = aws_internet_gateway.stock-rl-igw.id
# }

# Create a Public Subnet
resource "aws_subnet" "stock-rl-public-subnet" {
  vpc_id            = aws_vpc.stock-rl-vpc.id
  cidr_block        = "10.0.128.0/17"
  availability_zone = "us-east-1a"


  tags = {
    Name = "stock-rl-public-subnet"
  }
}

# Create a Route Table 
resource "aws_route_table" "stock-rl-public-route-table" {
  vpc_id = aws_vpc.stock-rl-vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.stock-rl-igw.id
  }

  tags = {
    Name = "stock-rl-public-route-table"
  }
}

# Create a Route Table Association
resource "aws_route_table_association" "stock-rl-public-route-table-association" {
  subnet_id      = aws_subnet.stock-rl-public-subnet.id
  route_table_id = aws_route_table.stock-rl-public-route-table.id
}

# Create a Security Group
resource "aws_security_group" "stock-rl-security-group" {
  name        = "stock-rl-security-group"
  description = "Allow inbound / outbound traffic from the Internet"
  vpc_id      = aws_vpc.stock-rl-vpc.id

  ingress {
    description = "Only Allow HTTPS from the Internet"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Uncomment and add your IP address to allow SSH access to the EC2 instance
  #   ingress {
  #     description = "Allow SSH Access from your personal computer"
  #     from_port   = 22
  #     to_port     = 22
  #     protocol    = "tcp"
  #     cidr_blocks = [] # Add your IP address here in format "0.0.0.0/32"
  #   }

  egress = [{
    description      = "Allow all outbound traffic"
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
    prefix_list_ids  = []
    security_groups  = []
    self             = false
  }]

  tags = {
    Name = "stock-rl-security-group"
  }
}


# # Create a network interface
# resource "aws_network_interface" "stock-rl-network-interface" {
#   subnet_id       = aws_subnet.stock-rl-public-subnet.id
#   private_ip      = "10.0.128.100"
#   security_groups = [aws_security_group.stock-rl-security-group.id]
# }

# # Create an EIP
# resource "aws_eip" "stock-rl-eip" {
#   vpc                       = true
#   network_interface         = aws_network_interface.stock-rl-network-interface.id
#   associate_with_private_ip = "10.0.128.100"
#   depends_on = [
#     aws_internet_gateway.stock-rl-igw
#   ]
# }

# # Create an EC2 Instance
# resource "aws_instance" "stock-rl-ec2-instance" {
#   ami                         = "ami-026b57f3c383c2eec"
#   instance_type               = "t2.micro"
#   availability_zone           = "us-east-1a"
#   key_name                    = "stock-rl-key-pair"
#   subnet_id                   = aws_subnet.stock-rl-public-subnet.id
#   associate_public_ip_address = true
#   vpc_security_group_ids      = [aws_security_group.stock-rl-security-group.id]
#   monitoring                  = true
#   network_interface {
#     network_interface_id = aws_network_interface.stock-rl-network-interface.id
#     device_index         = 0
#   }
#   # user_data = <<-EOF
#   # #! /bin/bash
#   #   sudo apt update -y
#   #   will fill in later once we know exactly what we need
#   #   EOF

#   tags = {
#     Name = "stock-rl-ec2-instance"
#   }
# }

# # Create an SNS topic to send notifications when trades are made
# resource "aws_sns_topic" "stock-rl-sns-topic" {
#   name = "stock-rl-sns-topic"
# }

# # Create an SNS topic subscription to send notifications to my email
# resource "aws_sns_topic_subscription" "stock-rl-sns-topic-subscription" {
#   topic_arn = aws_sns_topic.stock-rl-sns-topic.arn
#   protocol  = "email"
#   endpoint  = "your email address"
#   depends_on = [
#     aws_sns_topic.stock-rl-sns-topic
#   ]
# }