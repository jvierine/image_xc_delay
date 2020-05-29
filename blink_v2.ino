// (c) 2020 Juha Vierinen
// Camera timing calibrator signal generator
// Use these two pins to drive LEDs
int outPin1 = 13;      // connect this pin to led1
int outPin2 = 12;      // connect this pin to led2

// This is the basic delay unit
// a number between 10 and 10000 microseconds
int delta_pulse_length = 1000;
// How many delay units is one light pulse 
// e.g., if n_pulses=10 and delta_pulse_length=500, 
// then the pulse length is 5 ms. 
int min_n_pulses = 10;
int max_n_pulses = 200;
// how many delay units of time delay is there between led 1 and led 2
// in this case 5 would be 5 ms
int delay_deltas=5;

int bit0=1;

void setup()
{
  pinMode(outPin1, OUTPUT);    // sets the digital pin as output
  pinMode(outPin2, OUTPUT);    // sets the digital pin as output
  noInterrupts();              // disable interrupts to improve timing accuracy
}

// create a delay using delayMicroseconds
void delay_us2(int delta_pulse_length, int n_pulses)
{
 for (int i=0; i < n_pulses ; i++)
 {
    delayMicroseconds(delta_pulse_length);      
  }
}

void loop() 
{
  int n_pulses = random(min_n_pulses,max_n_pulses);  // randomize pulse length
  int dither = random(5,15) // dither the time delay a bit to reduce sampling issues
  digitalWrite(outPin1, bit0); // sets the pin to state
  delay_us2(delta_pulse_length, delay_deltas);
  delayMicroseconds(dither);
  digitalWrite(outPin2, bit0); // sets the pin to state
  delay_us2(delta_pulse_length, n_pulses-delay_deltas);
  bit0=(bit0+1)%2;		
}
