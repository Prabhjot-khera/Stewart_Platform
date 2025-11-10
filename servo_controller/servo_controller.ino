#include <Servo.h>

// ----- Servo configuration -----
Servo servo1;
Servo servo2;
Servo servo3;

const int servoPin1 = 9;
const int servoPin2 = 10;
const int servoPin3 = 11;

const int MIN_ANGLE = 0;
const int MAX_ANGLE = 30;
const int NEUTRAL_ANGLE = 15;

// Communication variables
int targetAngle1 = NEUTRAL_ANGLE;
int targetAngle2 = NEUTRAL_ANGLE;
int targetAngle3 = NEUTRAL_ANGLE;
bool newCommand = false;

// Byte buffer for reading 3 angles
byte angleBuffer[3];
int bytesReceived = 0;

void setup() {
  Serial.begin(9600);  // Match config.json baud rate

  servo1.attach(servoPin1);
  servo2.attach(servoPin2);
  servo3.attach(servoPin3);

  // Move all servos to neutral
  servo1.write(NEUTRAL_ANGLE);
  servo2.write(NEUTRAL_ANGLE);
  servo3.write(NEUTRAL_ANGLE);

  Serial.println("3-servo controller ready");
}

void loop() {
  // Read 3 bytes: angle1, angle2, angle3 (matching ballBalance copy method)
  while (Serial.available() > 0 && bytesReceived < 3) {
    angleBuffer[bytesReceived] = Serial.read();
    bytesReceived++;
  }

  // When we have all 3 bytes, process them
  if (bytesReceived >= 3) {
    int angle1 = angleBuffer[0];
    int angle2 = angleBuffer[1];
    int angle3 = angleBuffer[2];
    
    // Validate and clamp angles
    if (angle1 >= MIN_ANGLE && angle1 <= MAX_ANGLE &&
        angle2 >= MIN_ANGLE && angle2 <= MAX_ANGLE &&
        angle3 >= MIN_ANGLE && angle3 <= MAX_ANGLE) {
      targetAngle1 = angle1;
      targetAngle2 = angle2;
      targetAngle3 = angle3;
      newCommand = true;
    }
    
    // Reset buffer for next command
    bytesReceived = 0;
  }

  // Update servos if new command received
  if (newCommand) {
    servo1.write(targetAngle1);
    servo2.write(targetAngle2);
    servo3.write(targetAngle3);
    
    newCommand = false;
  }

  delay(10);
}