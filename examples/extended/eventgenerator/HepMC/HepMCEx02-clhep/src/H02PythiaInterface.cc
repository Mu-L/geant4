//
// ********************************************************************
// * DISCLAIMER                                                       *
// *                                                                  *
// * The following disclaimer summarizes all the specific disclaimers *
// * of contributors to this software. The specific disclaimers,which *
// * govern, are listed with their locations in:                      *
// *   http://cern.ch/geant4/license                                  *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.                                                             *
// *                                                                  *
// * This  code  implementation is the  intellectual property  of the *
// * GEANT4 collaboration.                                            *
// * By copying,  distributing  or modifying the Program (or any work *
// * based  on  the Program)  you indicate  your  acceptance of  this *
// * statement, and all its terms.                                    *
// ********************************************************************

// ====================================================================
//
//   H02PythiaInterface.cc
//   $Id: H02PythiaInterface.cc,v 1.2 2003/06/16 16:48:53 gunter Exp $
//
// ====================================================================
#include "H02PythiaInterface.hh"
#include "H02PythiaMessenger.hh"

#include "CLHEP/HepMC/CBhepevt.h"
#include "CLHEP/HepMC/include/PythiaWrapper6_2.h"

// additional pythia calls
#define pygive pygive_
#define pyrget pyrget_
#define pyrset pyrset_

extern "C" {
  void pygive(const char*, int);
  void pyrget(int*, int*);
  void pyrset(int*, int*);
}

void call_pygive(G4String s) { pygive(s.c_str(), s.length()); }
void call_pyrget(int a, int b) { pyrget(&a, &b); }
void call_pyrset(int a, int b) { pyrset(&a, &b); }

// global parameters
hepev2_t hepev2_;
hepev3_t hepev3_;
hepev4_t hepev4_;
hepev5_t hepev5_;

////////////////////////////////////////
H02PythiaInterface::H02PythiaInterface()
  : verbose(0), mpylist(0)
////////////////////////////////////////
{
  hepevtio= new HepMC::CBhepevt;
  messenger= new H02PythiaMessenger(this);
}

/////////////////////////////////////////
H02PythiaInterface::~H02PythiaInterface()
/////////////////////////////////////////
{
  delete hepevtio;
  delete messenger;
}

/////////////////////////////////////////////////
void H02PythiaInterface::CallPygive(G4String par)
/////////////////////////////////////////////////
{
  call_pygive(par);
}

//////////////////////////////////////////////////////////////////////
void H02PythiaInterface::CallPyinit(G4String frame, G4String beam, 
					G4String target, G4double win)
//////////////////////////////////////////////////////////////////////
{
  call_pyinit(frame.c_str(), beam.c_str(), target.c_str(), win);
}

////////////////////////////////////////////////
void H02PythiaInterface::CallPystat(G4int istat)
////////////////////////////////////////////////
{
  call_pystat(istat);
}


//////////////////////////////////////////////////////////
void H02PythiaInterface::CallPyrget(G4int lun, G4int move)
//////////////////////////////////////////////////////////
{
  call_pyrget(lun, move);
}

//////////////////////////////////////////////////////////
void H02PythiaInterface::CallPyrset(G4int lun, G4int move)
//////////////////////////////////////////////////////////
{
  call_pyrset(lun, move);
}


/////////////////////////////////////
void H02PythiaInterface::CallPyevnt()
/////////////////////////////////////
{
  call_pyevnt();
}

///////////////////////////////////////////////
void H02PythiaInterface::CallPylist(G4int mode)
///////////////////////////////////////////////
{
  call_pylist(mode);
}

///////////////////////////////////////////////
void H02PythiaInterface::CallPyhepc(G4int mode)
///////////////////////////////////////////////
{
  call_pyhepc(mode);
}


///////////////////////////////////////////////////
void H02PythiaInterface::SetRandomSeed(G4int iseed)
///////////////////////////////////////////////////
{
  pydatr.mrpy[1-1]= iseed;
}

//////////////////////////////////////////////////////////////////////
void H02PythiaInterface::PrintRandomStatus(std::ostream& ostr) const
//////////////////////////////////////////////////////////////////////
{
  ostr << "# Pythia random numbers status" << G4endl;
  for (G4int j=0; j<6; j++) {
    ostr << "pydatr.mrpy[" << j << "]= " << pydatr.mrpy[j] << G4endl;
  }
  for (G4int k=0; k<100; k++) {
    ostr << "pydatr.rrpy[" << k << "]= " << pydatr.rrpy[k] << G4endl;
  }
}

////////////////////////////////////////////
void H02PythiaInterface::SetUserParameters()
////////////////////////////////////////////
{
  G4cout << "set user parameters of PYTHIA common." << G4endl
	 << "nothing to be done in default."
	 << G4endl;
}

/////////////////////////////////////////////////////////
HepMC::GenEvent* H02PythiaInterface::GenerateHepMCEvent()
/////////////////////////////////////////////////////////
{
  static G4int nevent= 0; // event counter

  CallPyevnt(); // generate one event with Pythia
  if(mpylist >=1 && mpylist<= 3) CallPylist(mpylist);
  
  CallPyhepc(1);

  HepMC::GenEvent* evt= new HepMC::GenEvent();
  hepevtio-> toGenEvent(evt, false);
  hepevtio-> clean(); // clear HEPEVT common
  evt-> set_event_number(nevent++);
  if(verbose>0) evt-> print();

  return evt;
}

//////////////////////////////////////
void H02PythiaInterface::Print() const
//////////////////////////////////////
{
  G4cout << "PythiaInterface::Print()..." << G4endl;
}
