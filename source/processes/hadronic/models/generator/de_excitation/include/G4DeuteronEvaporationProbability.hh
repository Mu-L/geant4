// This code implementation is the intellectual property of
// the RD44 GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
//
// Hadronic Process: Nuclear De-excitations
// by V. Lara (Nov 1999) 
//



#ifndef G4DeuteronEvaporationProbability_h
#define G4DeuteronEvaporationProbability_h 1


#include "G4EvaporationProbability.hh"


class G4DeuteronEvaporationProbability : public G4EvaporationProbability
{
public:
	// Only available constructor
	G4DeuteronEvaporationProbability();

	~G4DeuteronEvaporationProbability() {}
private:  
	// Copy constructor
	G4DeuteronEvaporationProbability(const G4DeuteronEvaporationProbability &right);

	const G4DeuteronEvaporationProbability & operator=(const G4DeuteronEvaporationProbability &right);
	G4bool operator==(const G4DeuteronEvaporationProbability &right) const;
	G4bool operator!=(const G4DeuteronEvaporationProbability &right) const;
  

private:

	virtual G4double CalcAlphaParam(const G4Fragment & fragment) const 
	{ return 1.0 + CCoeficient(G4double(fragment.GetZ()-GetZ()));}
	
	virtual G4double CalcBetaParam(const G4Fragment & fragment) const 
	{ return 0.0; }

	
	G4double CCoeficient(const G4double aZ) const;

	// Excitation energy levels 
	G4RWTValVector<G4double> ExcitEnergies;
	// Spin of excitation energy levels 
	G4RWTValVector<G4int> ExcitSpins;

};
#endif
