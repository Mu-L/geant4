//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// 15.03.2016 V.Ivanchenko 
//
// List of parameters of the pre-compound model
// and the deexcitation module
//

#include "G4DeexPrecoParameters.hh"
#include "G4ApplicationState.hh"
#include "G4StateManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4PhysicsModelCatalog.hh"
#include "G4DeexParametersMessenger.hh"
#include "G4HadronicParameters.hh"
#include "G4Threading.hh"

G4DeexPrecoParameters::G4DeexPrecoParameters() 
{
  fStateManager = G4StateManager::GetStateManager();
  theMessenger = new G4DeexParametersMessenger(this);
  Initialise();
}

G4DeexPrecoParameters::~G4DeexPrecoParameters() 
{
  delete theMessenger;
}

void G4DeexPrecoParameters::SetDefaults()
{
  if(!IsLocked()) { Initialise(); }
}

void G4DeexPrecoParameters::Initialise()
{
  // common parameters
  fVerbose = 1;
  fLevelDensity = 0.075/CLHEP::MeV;
  fR0 = 1.5*CLHEP::fermi;
  fTransitionsR0 = 0.6*CLHEP::fermi;

  // preco parameters
  fPrecoLowEnergy = 0.1*CLHEP::MeV;
  fPrecoHighEnergy = 30*CLHEP::MeV;
  fPhenoFactor = 1.0;

  fPrecoType = 1;
  fMinZForPreco = 3;
  fMinAForPreco = 5;

  fNeverGoBack = false;
  fUseSoftCutoff = false;
  fUseCEM = true;
  fUseGNASH = false;
  fUseHETC = false;
  fUseAngularGen = true;
  fPrecoDummy = false;

  // de-exitation parameters 
  fMinExcitation = 10*CLHEP::eV;
  fNuclearLevelWidth = 0.2*CLHEP::MeV;
  fFBUEnergyLimit = 20.0*CLHEP::MeV; 
  fFermiEnergy = 35.0*CLHEP::MeV; 
  fMaxLifeTime = 1*CLHEP::nanosecond;
  fMinExPerNucleounForMF = 200*CLHEP::GeV;

  fDeexChannelType = fCombined;
  fPreCompoundType = eDefault;
  fFermiBreakUpType = bModelVI;
  fDeexType = 3;
  fTwoJMAX = 10;

  fCorrelatedGamma = false;
  fStoreAllLevels = true;
  fInternalConversion = true;
  fLD = true;  // use simple level density model 
  fFD = false; // use transition to discrete level 
  fIsomerFlag = true; // enable isomere production
}

void G4DeexPrecoParameters::SetLevelDensity(G4double val)
{
  if(IsLocked() || val <= 0.0) { return; }
  fLevelDensity = val/CLHEP::MeV;
}

void G4DeexPrecoParameters::SetR0(G4double val)
{
  if(IsLocked() || val <= 0.0) { return; }
  fR0 = val;
}

void G4DeexPrecoParameters::SetTransitionsR0(G4double val)
{
  if(IsLocked() || val <= 0.0) { return; }
  fTransitionsR0 = val;
}

void G4DeexPrecoParameters::SetFBUEnergyLimit(G4double val)
{
  if(IsLocked() || val <= 0.0) { return; }
  fFBUEnergyLimit = val;
}

void G4DeexPrecoParameters::SetFermiEnergy(G4double val)
{
  if(IsLocked() || val <= 0.0) { return; }
  fFermiEnergy = val;
}

void G4DeexPrecoParameters::SetPrecoLowEnergy(G4double val)
{
  if(IsLocked() || val < 0.0) { return; }
  fPrecoLowEnergy = val;
}

void G4DeexPrecoParameters::SetPrecoHighEnergy(G4double val)
{
  if(IsLocked() || val < 0.0) { return; }
  fPrecoHighEnergy = val;
}

void G4DeexPrecoParameters::SetPhenoFactor(G4double val)
{
  if(IsLocked() || val <= 0.0) { return; }
  fPhenoFactor = val;
}

void G4DeexPrecoParameters::SetMinExcitation(G4double val)
{
  if(IsLocked() || val < 0.0) { return; }
  fMinExcitation = val;
}

void G4DeexPrecoParameters::SetNuclearLevelWidth(G4double val)
{
  if(IsLocked() || val < 0.0) { return; }
  fNuclearLevelWidth = val;
}

void G4DeexPrecoParameters::SetMaxLifeTime(G4double val)
{
  if(IsLocked() || val < 0.0) { return; }
  fMaxLifeTime = val;
}

void G4DeexPrecoParameters::SetMinExPerNucleounForMF(G4double val)
{
  if(IsLocked() || val < 0.0) { return; }
  fMinExPerNucleounForMF = val;
}

void G4DeexPrecoParameters::SetMinZForPreco(G4int n)
{
  if(IsLocked() || n < 2) { return; }
  fMinZForPreco = n;
}

void G4DeexPrecoParameters::SetMinAForPreco(G4int n)
{
  if(IsLocked() || n < 0) { return; }
  fMinAForPreco = n;
}

void G4DeexPrecoParameters::SetPrecoModelType(G4int n)
{
  if(IsLocked() || n < 0 || n > 3) { return; }
  fPrecoType = n;
}

void G4DeexPrecoParameters::SetDeexModelType(G4int n)
{
  if(IsLocked() || n < 0 || n > 3) { return; }
  fDeexType = n;
}

void G4DeexPrecoParameters::SetTwoJMAX(G4int n)
{
  if(IsLocked() || n < 0) { return; }
  fTwoJMAX = n;
}

void G4DeexPrecoParameters::SetVerbose(G4int n)
{
  if(IsLocked()) { return; }
  fVerbose = n;
}

void G4DeexPrecoParameters::SetNeverGoBack(G4bool val)
{
  if(IsLocked()) { return; }
  fNeverGoBack = val;
}

void G4DeexPrecoParameters::SetUseSoftCutoff(G4bool val)
{
  if(IsLocked()) { return; }
  fUseSoftCutoff = val;
}

void G4DeexPrecoParameters::SetUseCEM(G4bool val)
{
  if(IsLocked()) { return; }
  fUseCEM = val;
}

void G4DeexPrecoParameters::SetUseGNASH(G4bool val)
{
  if(IsLocked()) { return; }
  fUseGNASH = val;
}

void G4DeexPrecoParameters::SetUseHETC(G4bool val)
{
  if(IsLocked()) { return; }
  fUseHETC = val;
}

void G4DeexPrecoParameters::SetUseAngularGen(G4bool val)
{
  if(IsLocked()) { return; }
  fUseAngularGen = val;
}

void G4DeexPrecoParameters::SetPrecoDummy(G4bool val)
{
  if(IsLocked()) { return; }
  fPrecoDummy = val;
  fDeexChannelType = fDummy;  
}

void G4DeexPrecoParameters::SetCorrelatedGamma(G4bool val)
{
  if(IsLocked()) { return; }
  fCorrelatedGamma = val; 
}

void G4DeexPrecoParameters::SetStoreICLevelData(G4bool val)
{
  if(IsLocked()) { return; }
  fStoreAllLevels = val;
}

void G4DeexPrecoParameters::SetStoreAllLevels(G4bool val)
{
  SetStoreICLevelData(val);
}

void G4DeexPrecoParameters::SetInternalConversionFlag(G4bool val)
{
  if(IsLocked()) { return; }
  fInternalConversion = val;
}

void G4DeexPrecoParameters::SetLevelDensityFlag(G4bool val)
{
  if(IsLocked()) { return; }
  fLD = val;
}

void G4DeexPrecoParameters::SetDiscreteExcitationFlag(G4bool val)
{
  if(IsLocked()) { return; }
  fFD = val;
}

void G4DeexPrecoParameters::SetIsomerProduction(G4bool val)
{
  if(IsLocked()) { return; }
  fIsomerFlag = val;
}

void G4DeexPrecoParameters::SetDeexChannelsType(G4DeexChannelType val)
{
  if(IsLocked()) { return; }
  fDeexChannelType = val;
}

void G4DeexPrecoParameters::SetPreCompoundType(G4PreCompoundType val)
{
  if(IsLocked()) { return; }
  fPreCompoundType = val;
}

void G4DeexPrecoParameters::SetFermiBreakUpType(G4FermiBreakUpType val)
{
  if(IsLocked()) { return; }
  fFermiBreakUpType = val;
}

std::ostream& G4DeexPrecoParameters::StreamInfo(std::ostream& os) const
{
  static const G4String namm[5] = {"Evaporation","GEM","Evaporation+GEM","GEMVI","Dummy"};
  static const G4int nmm[5] = {8, 68, 68, 83, 0};
  static const G4String nfbu[3] = {"ModelVI", "ModelAN", "Dummy"};
  G4int idx = fDeexChannelType;
  G4int jdx = fFermiBreakUpType;

  G4long prec = os.precision(5);
  os << "=======================================================================" << "\n";
  os << "======       Geant4 Native Pre-compound Model Parameters       ========" << "\n";
  os << "=======================================================================" << "\n";
  os << "Type of pre-compound model                          " << fPreCompoundType << "\n";
  os << "Type of pre-compound inverse x-section              " << fPrecoType << "\n";
  os << "Pre-compound model active                           " << (!fPrecoDummy) << "\n";
  os << "Pre-compound excitation low energy                  " 
     << G4BestUnit(fPrecoLowEnergy, "Energy") << "\n";
  os << "Pre-compound excitation high energy                 " 
     << G4BestUnit(fPrecoHighEnergy, "Energy") << "\n";
  os << "Angular generator for pre-compound model            " << fUseAngularGen << "\n";
  os << "Use NeverGoBack option for pre-compound model       " << fNeverGoBack << "\n";
  os << "Use SoftCutOff option for pre-compound model        " << fUseSoftCutoff << "\n";
  os << "Use CEM transitions for pre-compound model          " << fUseCEM << "\n";
  os << "Use GNASH transitions for pre-compound model        " << fUseGNASH << "\n";
  os << "Use HETC submodel for pre-compound model            " << fUseHETC << "\n";
  os << "=======================================================================" << "\n";
  os << "======       Nuclear De-excitation Module Parameters           ========" << "\n";
  os << "=======================================================================" << "\n";
  os << "Type of de-excitation inverse x-section             " << fDeexType << "\n";
  os << "Type of de-excitation factory                       " << namm[idx] << "\n";
  os << "Number of de-excitation channels                    " << nmm[idx] << "\n";
  os << "Type of Fermi BreakUp model                         " << nfbu[jdx] << "\n";
  os << "Min excitation energy                               " 
     << G4BestUnit(fMinExcitation, "Energy") << "\n";
  os << "Min energy per nucleon for multifragmentation       " 
     << G4BestUnit(fMinExPerNucleounForMF, "Energy") << "\n";
  os << "Limit excitation energy for Fermi BreakUp           " 
     << G4BestUnit(fFBUEnergyLimit, "Energy") << "\n";
  os << "Level density (1/MeV)                               " 
     << fLevelDensity*CLHEP::MeV << "\n";
  os << "Use simple level density model                      " << fLD << "\n";
  os << "Use discrete excitation energy of the residual      " << fFD << "\n";
  os << "Time limit for long lived isomeres                  " 
     << G4BestUnit(fMaxLifeTime, "Time") << "\n";
  os << "Isomer production flag                              " << fIsomerFlag << "\n";
  os << "Internal e- conversion flag                         " 
     << fInternalConversion << "\n";
  os << "Store e- internal conversion data                   " << fStoreAllLevels << "\n";
  os << "Correlated gamma emission flag                      " << fCorrelatedGamma << "\n";
  os << "Max 2J for sampling of angular correlations         " << fTwoJMAX << "\n";  
  os << "=======================================================================" << G4endl;
  os.precision(prec);
  return os;
}

G4int G4DeexPrecoParameters::GetVerbose() const
{
  G4int verb = G4HadronicParameters::Instance()->GetVerboseLevel();
  return (verb > 0) ? std::max(fVerbose, verb) : verb;
}

void G4DeexPrecoParameters::Dump()
{
  if(!fIsPrinted && GetVerbose() > 0 && G4Threading::IsMasterThread()) {
    StreamInfo(G4cout);
    fIsPrinted = true;
  }
}

std::ostream& operator<< (std::ostream& os, const G4DeexPrecoParameters& par)
{
  return par.StreamInfo(os);
}

G4bool G4DeexPrecoParameters::IsLocked() const
{
  return (!G4Threading::IsMasterThread() ||
	  (fStateManager->GetCurrentState() != G4State_PreInit));
}
