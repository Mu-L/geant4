// Split barion (antibarion) into quark and diquark (antidiquark and antiqaurk ) 
// based on prototype, needs clean up of interfaces HPW Feb 1999  
// Numbers verified and errors corrected, HPW Dec 1999

#include "G4BaryonSplitter.hh"
#include "G4ParticleTable.hh"

G4BaryonSplitter::
G4BaryonSplitter()
{
  theBaryons.insert(new G4SPBaryon(G4Proton::Proton()));
  theBaryons.insert(new G4SPBaryon(G4Neutron::Neutron()));
  theBaryons.insert(new G4SPBaryon(G4AntiProton::AntiProton()));
  theBaryons.insert(new G4SPBaryon(G4AntiNeutron::AntiNeutron()));
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(2224))); // D++
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(2214))); // D+
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(2114))); // D0
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(1114))); // D-
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(-2224))); // anti D++
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(-2214))); // anti D+
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(-2114))); // anti D0
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(-1114))); // anti D-
  theBaryons.insert(new G4SPBaryon(G4Lambda::Lambda()));
  theBaryons.insert(new G4SPBaryon(G4AntiLambda::AntiLambda()));
  theBaryons.insert(new G4SPBaryon(G4SigmaPlus::SigmaPlus()));
  theBaryons.insert(new G4SPBaryon(G4SigmaZero::SigmaZero()));
  theBaryons.insert(new G4SPBaryon(G4SigmaMinus::SigmaMinus()));
  theBaryons.insert(new G4SPBaryon(G4AntiSigmaPlus::AntiSigmaPlus()));
  theBaryons.insert(new G4SPBaryon(G4AntiSigmaZero::AntiSigmaZero()));
  theBaryons.insert(new G4SPBaryon(G4AntiSigmaMinus::AntiSigmaMinus()));
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(3224))); // S+*
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(3214))); // S0*
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(3114))); // S-*
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(-3224))); // anti S+*
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(-3214))); // anti S0*
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(-3114))); // anti S-*
  theBaryons.insert(new G4SPBaryon(G4XiMinus::XiMinus()));
  theBaryons.insert(new G4SPBaryon(G4XiZero::XiZero()));
  theBaryons.insert(new G4SPBaryon(G4AntiXiMinus::AntiXiMinus()));
  theBaryons.insert(new G4SPBaryon(G4AntiXiZero::AntiXiZero()));
  theBaryons.insert(new G4SPBaryon(G4OmegaMinus::OmegaMinus()));
  theBaryons.insert(new G4SPBaryon(G4AntiOmegaMinus::AntiOmegaMinus()));
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(3324))); // X0*
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(3314))); // X-*
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(-3324))); // anti X0*
  theBaryons.insert(new G4SPBaryon(G4ParticleTable::GetParticleTable()->FindParticle(-3314))); // anti X-*
}

G4bool G4BaryonSplitter::
SplitBarion(G4int PDGCode, G4int* q_or_qqbar, G4int* qbar_or_qq)
{
  const G4SPBaryon * aBaryon = theBaryons.GetBaryon(G4ParticleTable::GetParticleTable()->FindParticle(PDGCode));
  if(aBaryon==NULL)
  {
    return FALSE;
  }
  else
  {
    aBaryon->SampleQuarkAndDiquark(*q_or_qqbar, *qbar_or_qq);
    return TRUE;
  }
}


// Get the splittable baryon for a PDG code.
const G4SPBaryon & G4BaryonSplitter::
GetSPBaryon(G4int PDGCode)
{
return *theBaryons.GetBaryon(G4ParticleTable::GetParticleTable()->FindParticle(PDGCode));
}


// Find rest diquark in given barion after quark - antiquark annihilation  
G4bool G4BaryonSplitter::
FindDiquark(G4int PDGCode, G4int Quark, G4int* Diquark)
{
  const G4SPBaryon * aBaryon = theBaryons.GetBaryon(G4ParticleTable::GetParticleTable()->FindParticle(PDGCode));
  if(aBaryon==NULL)
  {
    return FALSE;
  }
  else
  {
    aBaryon->FindDiquark(Quark, *Diquark);
    return TRUE;
  }
}

