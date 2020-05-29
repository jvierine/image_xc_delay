/*
  Blink

  Turns an LED on for one second, then off for one second, repeatedly.

  Most Arduinos have an on-board LED you can control. On the UNO, MEGA and ZERO
  it is attached to digital pin 13, on MKR1000 on pin 6. LED_BUILTIN is set to
  the correct LED pin independent of which board is used.
  If you want to know what pin the on-board LED is connected to on your Arduino
  model, check the Technical Specs of your board at:
  https://www.arduino.cc/en/Main/Products

  modified 8 May 2014
  by Scott Fitzgerald
  modified 2 Sep 2016
  by Arturo Guadalupi
  modified 8 Sep 2016
  by Colby Newman

  This example code is in the public domain.

  http://www.arduino.cc/en/Tutorial/Blink
*/

int led1 = 13;
int led2 = 12;
int led3 = 11;
// the setup function runs once when you press reset or power the board


long t0;       // time when led goes on
long deltat;   // how long is the led on and off
long tau;      // relative time delay between l1 and l2
int done;
long min_deltat;
long max_deltat;
long tnow;
int l1_on;
int l2_on;

void setup() {
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(led1, OUTPUT);
  pinMode(led2, OUTPUT);
  pinMode(led3, OUTPUT);
  //Serial.begin(9600);
  done=0;
  tau=100000;
  min_deltat=5000;
  max_deltat= 100000;
  l1_on=0;
  l2_on=0;
}


// the loop function runs over and over again forever
void loop() 
{  
  if(done == 0)
  {
    t0=micros();
    deltat = random(min_deltat,max_deltat);
  //  Serial.println("t0");
  //  Serial.println(t0);
  //  Serial.println("deltat");
  //  Serial.println(deltat);
    done=1;
  }
  tnow = micros();
  long time_delta = tnow-t0;
//  Serial.println(time_delta);
  if( (time_delta > 0) && (time_delta < deltat) && (l1_on == 0)){
    digitalWrite(led1, HIGH);   // turn the LED on (HIGH is the voltage level)  
    l1_on=1;
  } 
  if( (time_delta > deltat) && (time_delta < 2*deltat) && (l1_on == 1)){
    digitalWrite(led1, LOW);   // turn the LED on (HIGH is the voltage level)  
    l1_on=0;
  } 
  
  if( (time_delta > tau) && (time_delta < (deltat+tau)) && (l2_on == 0)){
    digitalWrite(led2, HIGH);   // turn the LED on (HIGH is the voltage level)  
    l2_on=1;
  } 
  if( (time_delta > (deltat+tau)) && (time_delta < (2*deltat+tau)) && (l2_on == 1)){
    digitalWrite(led2, LOW);   // turn the LED on (HIGH is the voltage level)  
    l2_on=0;
  } 

  
  if( (l2_on == 0) && (time_delta > (2*deltat+tau)) ){
    done=0;
  }
}
