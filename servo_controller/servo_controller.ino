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

// Buffer for reading serial data
String inputString = "";
bool stringComplete = false;

void setup() {
  Serial.begin(115200);  // Match config.json baud rate

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
  // Read serial data (expecting format: "angle1,angle2,angle3\n")
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    
    if (inChar == '\n') {
      stringComplete = true;
      break;
    } else {
      inputString += inChar;
    }
  }

  // Parse the received string
  if (stringComplete) {
    // Parse format: "angle1,angle2,angle3"
    int comma1 = inputString.indexOf(',');
    int comma2 = inputString.indexOf(',', comma1 + 1);
    
    if (comma1 > 0 && comma2 > comma1) {
      int angle1 = inputString.substring(0, comma1).toInt();
      int angle2 = inputString.substring(comma1 + 1, comma2).toInt();
      int angle3 = inputString.substring(comma2 + 1).toInt();
      
      // Validate and clamp angles
      if (angle1 >= MIN_ANGLE && angle1 <= MAX_ANGLE &&
          angle2 >= MIN_ANGLE && angle2 <= MAX_ANGLE &&
          angle3 >= MIN_ANGLE && angle3 <= MAX_ANGLE) {
        targetAngle1 = angle1;
        targetAngle2 = angle2;
        targetAngle3 = angle3;
        newCommand = true;
      }
    }
    
    // Clear the string for next input
    inputString = "";
    stringComplete = false;
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