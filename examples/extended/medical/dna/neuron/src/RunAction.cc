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
// This example is provided by the Geant4-DNA collaboration
// Any report or published results obtained using the Geant4-DNA software
// shall cite the following Geant4-DNA collaboration publication:
// Med. Phys. 37 (2010) 4692-4708
// and papers
// M. Batmunkh et al. J Radiat Res Appl Sci 8 (2015) 498-507
// O. Belov et al. Physica Medica 32 (2016) 1510-1520
// The Geant4-DNA web site is available at http://geant4-dna.org
//
// -------------------------------------------------------------------
// November 2016
// -------------------------------------------------------------------
//
/// \file RunAction.cc
/// \brief Implementation of the RunAction class

#include "RunAction.hh"

#include "CommandLineParser.hh"
#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "Run.hh"
#include "TrackingAction.hh"

#include "G4AnalysisManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4UImanager.hh"
#include "G4UnitsTable.hh"
#include "Randomize.hh"

#include <iomanip>

using namespace G4DNAPARSER;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::RunAction(DetectorConstruction* det, PrimaryGeneratorAction* prim)
  : G4UserRunAction(), fDetector(det), fPrimary(prim)
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4Run* RunAction::GenerateRun()
{
  if (nullptr == fRun) {
    fRun = new Run(fDetector);
  }
  return fRun;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::BeginOfRunAction(const G4Run* /*run*/)
{
  RunInitManager::Instance()->Initialize();

  if (nullptr != fPrimary) {
    G4ParticleDefinition* particle = fPrimary->GetParticleGun()->GetParticleDefinition();
    G4double energy = fPrimary->GetParticleGun()->GetParticleEnergy();
    fRun->SetPrimary(particle, energy);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::EndOfRunAction(const G4Run* /*run*/)
{
  if (isMaster) fRun->EndOfRun();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::CreateHistogram()
{
  // Book histograms, ntuple
  // Create analysis manager

  CommandLineParser* parser = CommandLineParser::GetParser();
  Command* command(0);
  if ((command = parser->GetCommandIfActive("-out")) == 0) return;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::WriteHistogram()
{
  CommandLineParser* parser = CommandLineParser::GetParser();
  Command* commandLine(0);
  if ((commandLine = parser->GetCommandIfActive("-out")) == 0) return;

  // print histogram statistics
  //
  G4AnalysisManager* analysisManager = G4AnalysisManager::Instance();

  // save histograms
  //
  analysisManager->Write();
  analysisManager->CloseFile();

  if (fDebug) {
    G4cout << "================ ROOT FILES HAVE BEEN WRITTEN" << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::PrintRunInfo(const G4Run* run)
{
  G4cout << "================ Run is = " << run->GetRunID() << G4endl;
  G4cout << "================ Run type is = " << G4RunManager::GetRunManager()->GetRunManagerType()
         << G4endl;
  G4cout << "================ Event processed = " << run->GetNumberOfEventToBeProcessed() << G4endl;
  G4cout << "================ Nevent = " << run->GetNumberOfEvent() << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
