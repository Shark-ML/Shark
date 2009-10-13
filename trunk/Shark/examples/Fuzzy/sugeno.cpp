
#include <iostream>
#include <Fuzzy/SigmoidalLT.h>
#include <Fuzzy/SugenoIM.h>
#include <Fuzzy/SugenoRule.h>
#include <Fuzzy/TriangularLT.h>


using namespace std;


int main()
{
// Die Regel soll diesmal sowas sein wie
// Wenn man (bei der Pruefung) nicht so richtig ausgeschlafen ist
// und obendrein noch hacke, so ist das Pruefungsergebnis 
// wahrscheinlich schlecht.

  // Definiere LV:
  //cout<<"Definiere LV:"<<endl;

  RCPtr<LinguisticVariable> AG( new LinguisticVariable( "Ausgeschlafenheit" ) );
  RCPtr<LinguisticVariable> N( new LinguisticVariable( "Nuechternheit" ) );
  RCPtr<LinguisticVariable> PE (new LinguisticVariable( "Pruefungsergebnis" ) );


  // Definiere Terme:
  //cout<<"Definiere Terme:"<<endl;

  RCPtr<LinguisticTerm> sehrAG(new SigmoidalLT("sehrAG",AG,3,9));// mehr als neun Stunden
  RCPtr<LinguisticTerm> maessigAG(new TriangularLT("maessig ausgeschlafen",AG,7,9,10));//zwischen 7 und 10 std
  RCPtr<LinguisticTerm> muede(new SigmoidalLT("muede",AG,-1,7));//weniger als sieben Stunden

  RCPtr<LinguisticTerm> nuechtern(new SigmoidalLT("nuechtern",N,-2,0.5)); 
  //Ein Blutalkoholwerte von 0.5 markiert die Grenze
  RCPtr<LinguisticTerm> betrunken(new SigmoidalLT("betrunken",N,2,0.5));

  // Die Punkteskala reiche von 0 (schlechtest) bis 20 (best)

  // Gebe Plotdaten der Terme aus

  //cout<<"Gebe Plotdaten der Terme aus"<<endl;
  sehrAG->makeGNUPlotData("sehrAG.dat",500,-5,30);
  maessigAG->makeGNUPlotData("maessigAG.dat",200);
  muede->makeGNUPlotData("muede.dat",500,-10,20);
  nuechtern->makeGNUPlotData("nuechtern.dat",500,-3,3);
  betrunken->makeGNUPlotData("betrunken.dat",500,-3,3); 

  // Definiere Regelbasis:

  //cout<<"Definiere Regelbasis:"<<endl;
  RuleBase myBase;
  myBase.addToInputFormat(AG,N);
  myBase.addToOutputFormat(PE);
  
  // Definiere Regeln:
  //cout<<"Definiere Regeln:"<<endl;

  RCPtr<SugenoRule> Rfrisch(new SugenoRule(AND,&myBase));
  Rfrisch->addPremise(sehrAG, nuechtern);
  Rfrisch->setConclusion(5.0,1.5,-10.0);

  RCPtr<SugenoRule> Rmittel(new SugenoRule(AND,&myBase));
  Rmittel->addPremise(maessigAG,nuechtern);
  Rmittel->setConclusion(1.5,2.0,-9.0);

  RCPtr<SugenoRule> Rfertig(new SugenoRule(OR,&myBase));
  Rfertig->addPremise(muede, betrunken);
  Rfertig->setConclusion(-2.0,2.0,-3.0);

  // Definiere Inferenzmaschine
  //cout<<"Definiere Inferenzmaschine"<<endl;

  SugenoIM myMachine(&myBase);
  myMachine.characteristicCurve("c.dat", 20);

  // Berechne Ergebnisse fuer Beispiel-Inputvektoren
  cout<<"Berechne Ergebnisse fuer Beispiel Input-Vektoren"<<endl;

  std::vector<double> v( 2 );
  v[0] = 9.0;
  v[1] = 0.3;
  double Ergebnis1 = myMachine.computeSugenoInference(v);
  v[0] = 7.0;
  v[1] = 0.4;
  double Ergebnis2 = myMachine.computeSugenoInference(v);
  v[0] = 5.0;
  v[1] = 0.0;
  double Ergebnis3 = myMachine.computeSugenoInference(v);
  v[0] = 2.0;
  v[1] = 30.0;
  double Ergebnis4 = myMachine.computeSugenoInference(v);
  v[0] = -100.0;
  v[1] = -900;
  double Ergebnis5 = myMachine.computeSugenoInference(v);

  // Gebe die Ergebnisse aus:
  //   cout<<"Gebe die Ergebnisse aus:"<<endl;


//     cout<<"Rfrisch.Activation(): "<<Rfrisch->Activation(in1)<<endl;
//     cout<<"Rmimttel.Activation(): "<<Rmittel->Activation(in1)<<endl;
//     cout<<"Rfertig.Activation(): "<<Rfertig->Activation(in1)<<endl;

//     cout<<"Rfrisch.calculateConsequence() "<<Rfrisch.calculateConsequence(in1)<<endl;
//     cout<<"Rmittel.calculateConsequence() "<<Rmittel.calculateConsequence(in1)<<endl;
//     cout<<"Rfertig.calculateConsequence() "<<Rfertig.calculateConsequence(in1)<<endl;


     cout<<"Ergebnisse: "<<endl
         <<"Fuer Wert Nummer 1:" << Ergebnis1<<endl
         <<"Fuer Wert Nummer 2:" << Ergebnis2<<endl
         <<"Fuer Wert Nummer 3:" << Ergebnis3<<endl
         <<"Fuer Wert Nummer 4:" << Ergebnis4<<endl
         <<"Fuer Wert Nummer 5:" << Ergebnis5<<endl;

  // Teste die Ergebnisse an vorberechneten Punkten:
  //cout<<"Teste die Ergebnisse"<<endl;
  bool richtigGerechnet;

  richtigGerechnet = 
     int(1E3*Ergebnis1)==15911 &&   
     // nur dieses erste hier habe ich per Hand nachgerechnet
     // alle weiteren dienen dazu
     // 1.) nochmal durchgerechnet zu werden
     // 2.) Veraenderungen anzuzeigen.
     int(1E3*Ergebnis2)==10803 &&
     int(1E3*Ergebnis3)==8000 &&
     int(Ergebnis4)== -88 &&		// Fuer Volltrunkenheit gibt es also eine -88
     int(Ergebnis5)== 2498;

	return(richtigGerechnet);
}
