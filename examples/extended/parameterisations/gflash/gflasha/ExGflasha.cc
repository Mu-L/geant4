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
//
/// \file ExGflasha.cc
/// \brief Main program of the parameterisations/gflash/gflasha example
//

// G4 includes
#include "G4PhysListFactory.hh"
#include "G4RunManagerFactory.hh"
#include "G4Timer.hh"
#include "G4Types.hh"
#include "G4UImanager.hh"
#include "G4ios.hh"

// my project
#include "ExGflashActionInitialization.hh"
#include "ExGflashDetectorConstruction.hh"

#include "G4FastSimulationPhysics.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

int main(int argc, char** argv)
{
  // Instantiate G4UIExecutive if interactive mode
  G4UIExecutive* ui = nullptr;
  if (argc == 1) {
    ui = new G4UIExecutive(argc, argv);
  }

  // timer to see GFlash performance
  G4Timer timer;
  timer.Start();

  G4cout << "+-------------------------------------------------------+" << G4endl;
  G4cout << "|                                                       |" << G4endl;
  G4cout << "|          This is an example of Shower                 |" << G4endl;
  G4cout << "|          Parameterization with GFLASH                 |" << G4endl;
  G4cout << "+-------------------------------------------------------+" << G4endl;

  auto* runManager = G4RunManagerFactory::CreateRunManager();

  // UserInitialization classes (mandatory)
  auto detector = new ExGflashDetectorConstruction;
  runManager->SetUserInitialization(detector);

  // -- Select a physics list:
  G4PhysListFactory listFactory;
  G4String name;
  auto list_name = std::getenv("PHYSLIST");
  if (list_name == nullptr || std::strlen(list_name) == 0) {
    name = "FTFP_BERT";
  }
  else {
    name = list_name;
  }
  G4VModularPhysicsList* physicsList = listFactory.GetReferencePhysList(name);

  // -- Create a fast simulation physics constructor, used to augment
  // -- the above physics list to allow for fast simulation:
  auto fastSimulationPhysics = new G4FastSimulationPhysics();

  // -- We now configure the fastSimulationPhysics object.
  // -- The gflash model (GFlashShowerModel, see
  // ExGflashDetectorConstruction.cc)
  // -- is applicable to e+ and e- : we augment the physics list for these
  // -- particles (by adding a G4FastSimulationManagerProcess with below's
  // -- calls), this will make the fast simulation to be activated:
  fastSimulationPhysics->ActivateFastSimulation("e-");
  fastSimulationPhysics->ActivateFastSimulation("e+");
  // -- Register this fastSimulationPhysics to the physicsList,
  // -- when the physics list will be called by the run manager
  // -- (will happen at initialization of the run manager)
  // -- for physics process construction, the fast simulation
  // -- configuration will be applied as well.
  physicsList->RegisterPhysics(fastSimulationPhysics);
  runManager->SetUserInitialization(physicsList);

  // Action initialization:
  runManager->SetUserInitialization(new ExGflashActionInitialization(detector));

  G4VisManager* visManager = new G4VisExecutive;
  visManager->Initialize();

  G4UImanager* UImanager = G4UImanager::GetUIpointer();

  if (ui != nullptr)  // Define UI terminal for interactive mode
  {
    UImanager->ApplyCommand("/control/execute vis.mac");
    ui->SessionStart();
    delete ui;
  }
  else  // Batch mode
  {
    G4String s = *(argv + 1);
    UImanager->ApplyCommand("/control/execute " + s);
  }

  delete visManager;
  delete runManager;

  timer.Stop();
  G4cout << G4endl;
  G4cout << "******************************************";
  G4cout << G4endl;
  G4cout << "Total Real Elapsed Time is: " << timer.GetRealElapsed();
  G4cout << G4endl;
  G4cout << "Total System Elapsed Time: " << timer.GetSystemElapsed();
  G4cout << G4endl;
  G4cout << "Total GetUserElapsed Time: " << timer.GetUserElapsed();
  G4cout << G4endl;
  G4cout << "******************************************";
  G4cout << G4endl;

  return 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
