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
#ifdef USE_INFERENCE
#  include "Par04InferenceMessenger.hh"

#  include "Par04InferenceSetup.hh"  // for Par04InferenceSetup

#  include "G4UIcmdWithADoubleAndUnit.hh"  // for G4UIcmdWithADoubleAndUnit
#  include "G4UIcmdWithAString.hh"  // for G4UIcmdWithAString
#  include "G4UIcmdWithAnInteger.hh"  // for G4UIcmdWithAnInteger
#  include "G4UIdirectory.hh"  // for G4UIdirectory

#  include <CLHEP/Units/SystemOfUnits.h>  // for pi
#  include <G4ApplicationState.hh>  // for G4State_Idle
#  include <G4ThreeVector.hh>  // for G4ThreeVector
#  include <G4UImessenger.hh>  // for G4UImessenger
#  include <string>  // for stoi
class G4UIcommand;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Par04InferenceMessenger::Par04InferenceMessenger(Par04InferenceSetup* aInference)
  : G4UImessenger(), fInference(aInference)
{
  fExampleDir = new G4UIdirectory("/Par04/");
  fExampleDir->SetGuidance("UI commands specific to this example");

  fInferenceDir = new G4UIdirectory("/Par04/inference/");
  fInferenceDir->SetGuidance("Inference construction UI commands");

  fInferenceLibraryCmd = new G4UIcmdWithAString("/Par04/inference/setInferenceLibrary", this);
  fInferenceLibraryCmd->SetGuidance("Inference library.");
  fInferenceLibraryCmd->SetParameterName("InferenceLibrary", false);
  fInferenceLibraryCmd->AvailableForStates(G4State_Idle);
  fInferenceLibraryCmd->SetToBeBroadcasted(true);

  fSizeLatentVectorCmd = new G4UIcmdWithAnInteger("/Par04/inference/setSizeLatentVector", this);
  fSizeLatentVectorCmd->SetGuidance("Set size of the latent space vector.");
  fSizeLatentVectorCmd->SetParameterName("SizeLatentVector", false);
  fSizeLatentVectorCmd->SetRange("SizeLatentVector>0");
  fSizeLatentVectorCmd->AvailableForStates(G4State_Idle);
  fSizeLatentVectorCmd->SetToBeBroadcasted(true);

  fSizeConditionVectorCmd =
    new G4UIcmdWithAnInteger("/Par04/inference/setSizeConditionVector", this);
  fSizeConditionVectorCmd->SetGuidance("Set size of the condition vector.");
  fSizeConditionVectorCmd->SetParameterName("SizeConditionVector", false);
  fSizeConditionVectorCmd->SetRange("SizeConditionVector>0");
  fSizeConditionVectorCmd->AvailableForStates(G4State_Idle);
  fSizeConditionVectorCmd->SetToBeBroadcasted(true);

  fModelPathNameCmd = new G4UIcmdWithAString("/Par04/inference/setModelPathName", this);
  fModelPathNameCmd->SetGuidance("Model path and name.");
  fModelPathNameCmd->SetParameterName("Name", false);
  fModelPathNameCmd->AvailableForStates(G4State_Idle);
  fModelPathNameCmd->SetToBeBroadcasted(true);

  fModelTypeCmd = new G4UIcmdWithAString("/Par04/inference/setModelType", this);
  fModelTypeCmd->SetGuidance("Model type");
  fModelTypeCmd->SetParameterName("Name", false);
  fModelTypeCmd->AvailableForStates(G4State_Idle);
  fModelTypeCmd->SetToBeBroadcasted(true);

  fProfileFlagCmd = new G4UIcmdWithAnInteger("/Par04/inference/setProfileFlag", this);
  fProfileFlagCmd->SetGuidance("Flag to save a json file for model execution profiling.");
  fProfileFlagCmd->SetParameterName("ProfileFlag", false);
  fProfileFlagCmd->SetRange("ProfileFlag>-1");
  fProfileFlagCmd->AvailableForStates(G4State_Idle);
  fProfileFlagCmd->SetToBeBroadcasted(true);

  fOptimizationFlagCmd = new G4UIcmdWithAnInteger("/Par04/inference/setOptimizationFlag", this);
  fOptimizationFlagCmd->SetGuidance("Set optimization flag");
  fOptimizationFlagCmd->SetParameterName("OptimizationFlag", false);
  fOptimizationFlagCmd->SetRange("OptimizationFlag>-1");
  fOptimizationFlagCmd->AvailableForStates(G4State_Idle);
  fOptimizationFlagCmd->SetToBeBroadcasted(true);

  fMeshNbRhoCellsCmd = new G4UIcmdWithAnInteger("/Par04/inference/setNbOfRhoCells", this);
  fMeshNbRhoCellsCmd->SetGuidance("Set number of rho cells in the cylindrical mesh readout.");
  fMeshNbRhoCellsCmd->SetParameterName("NbRhoCells", false);
  fMeshNbRhoCellsCmd->SetRange("NbRhoCells>0");
  fMeshNbRhoCellsCmd->AvailableForStates(G4State_Idle);
  fMeshNbRhoCellsCmd->SetToBeBroadcasted(true);

  fMeshNbPhiCellsCmd = new G4UIcmdWithAnInteger("/Par04/inference/setNbOfPhiCells", this);
  fMeshNbPhiCellsCmd->SetGuidance("Set number of phi cells in the cylindrical mesh readout.");
  fMeshNbPhiCellsCmd->SetParameterName("NbPhiCells", false);
  fMeshNbPhiCellsCmd->SetRange("NbPhiCells>0");
  fMeshNbPhiCellsCmd->AvailableForStates(G4State_Idle);
  fMeshNbPhiCellsCmd->SetToBeBroadcasted(true);

  fMeshNbZCellsCmd = new G4UIcmdWithAnInteger("/Par04/inference/setNbOfZCells", this);
  fMeshNbZCellsCmd->SetGuidance("Set number of z cells in the cylindrical mesh readout.");
  fMeshNbZCellsCmd->SetParameterName("NbZCells", false);
  fMeshNbZCellsCmd->SetRange("NbZCells>0");
  fMeshNbZCellsCmd->AvailableForStates(G4State_Idle);
  fMeshNbZCellsCmd->SetToBeBroadcasted(true);

  fMeshSizeRhoCellsCmd = new G4UIcmdWithADoubleAndUnit("/Par04/inference/setSizeOfRhoCells", this);
  fMeshSizeRhoCellsCmd->SetGuidance("Set size of rho cells in the cylindrical readout mesh");
  fMeshSizeRhoCellsCmd->SetParameterName("Size", false);
  fMeshSizeRhoCellsCmd->SetRange("Size>0.");
  fMeshSizeRhoCellsCmd->SetUnitCategory("Length");
  fMeshSizeRhoCellsCmd->AvailableForStates(G4State_Idle);
  fMeshSizeRhoCellsCmd->SetToBeBroadcasted(true);

  fMeshSizeZCellsCmd = new G4UIcmdWithADoubleAndUnit("/Par04/inference/setSizeOfZCells", this);
  fMeshSizeZCellsCmd->SetGuidance("Set size of z cells in the cylindrical readout mesh");
  fMeshSizeZCellsCmd->SetParameterName("Size", false);
  fMeshSizeZCellsCmd->SetRange("Size>0.");
  fMeshSizeZCellsCmd->SetUnitCategory("Length");
  fMeshSizeZCellsCmd->AvailableForStates(G4State_Idle);
  fMeshSizeZCellsCmd->SetToBeBroadcasted(true);

  // Onnx Runtime Execution Provider flag commands
  fCudaFlagCmd = new G4UIcmdWithAnInteger("/Par04/inference/setCudaFlag", this);
  G4cout << "f CudaFlagCmd " << fCudaFlagCmd << G4endl;
  fCudaFlagCmd->SetGuidance("Whether to use CUDA Execution Provider for Onnx Runtime or not");
  fCudaFlagCmd->SetParameterName("CudaFlag", false);
  fCudaFlagCmd->SetRange("CudaFlag>-1");
  fCudaFlagCmd->AvailableForStates(G4State_Idle);
  fCudaFlagCmd->SetToBeBroadcasted(true);

  /// OnnxRuntime Execution Provider Options
  /// Cuda
  fCudaOptionsDir = new G4UIdirectory("/Par04/inference/cuda/");
  fCudaOptionsDir->SetGuidance("Commands for setting options for Cuda execution provider");

  fCudaDeviceIdCmd = new G4UIcmdWithAString("/Par04/inference/cuda/setDeviceId", this);
  fCudaDeviceIdCmd->SetGuidance("Device ID of Device on which to run CUDA code");
  fCudaDeviceIdCmd->SetParameterName("CudaDeviceId", false);
  fCudaDeviceIdCmd->AvailableForStates(G4State_Idle);
  fCudaDeviceIdCmd->SetToBeBroadcasted(true);

  fCudaGpuMemLimitCmd = new G4UIcmdWithAString("/Par04/inference/cuda/setGpuMemLimit", this);
  fCudaGpuMemLimitCmd->SetGuidance("GPU Memory limit for CUDA");
  fCudaGpuMemLimitCmd->SetParameterName("CudaGpuMemLimit", false);
  fCudaGpuMemLimitCmd->AvailableForStates(G4State_Idle);
  fCudaGpuMemLimitCmd->SetToBeBroadcasted(true);

  fCudaArenaExtendedStrategyCmd =
    new G4UIcmdWithAString("/Par04/inference/cuda/setArenaExtendedStrategy", this);
  fCudaArenaExtendedStrategyCmd->SetGuidance(
    "Strategy for extending the device memory arena for CUDA");
  fCudaArenaExtendedStrategyCmd->SetParameterName("CudaArenaExtendedStrategy", false);
  fCudaArenaExtendedStrategyCmd->AvailableForStates(G4State_Idle);
  fCudaArenaExtendedStrategyCmd->SetToBeBroadcasted(true);

  fCudaCudnnConvAlgoSearchCmd =
    new G4UIcmdWithAString("/Par04/inference/cuda/setCudnnConvAlgoSearch", this);
  fCudaCudnnConvAlgoSearchCmd->SetGuidance("Set which cuDNN Convolution Operation to use");
  fCudaCudnnConvAlgoSearchCmd->SetParameterName("CudaCudnnConvAlgoSearch", false);
  fCudaCudnnConvAlgoSearchCmd->AvailableForStates(G4State_Idle);
  fCudaCudnnConvAlgoSearchCmd->SetToBeBroadcasted(true);

  fCudaDoCopyInDefaultStreamCmd =
    new G4UIcmdWithAString("/Par04/inference/cuda/setDoCopyInDefaultStream", this);
  fCudaDoCopyInDefaultStreamCmd->SetGuidance("Whether to use same stream for copying");
  fCudaDoCopyInDefaultStreamCmd->SetParameterName("CudaDoCopyInDefaultStream", false);
  fCudaDoCopyInDefaultStreamCmd->AvailableForStates(G4State_Idle);
  fCudaDoCopyInDefaultStreamCmd->SetToBeBroadcasted(true);

  fCudaCudnnConvUseMaxWorkspaceCmd =
    new G4UIcmdWithAString("/Par04/inference/cuda/setCudnnConvUseMaxWorkspace", this);
  fCudaCudnnConvUseMaxWorkspaceCmd->SetGuidance("Memory Limit for cuDNN convolution operations");
  fCudaCudnnConvUseMaxWorkspaceCmd->SetParameterName("CudaCudnnConvUseMaxWorkspace", false);
  fCudaCudnnConvUseMaxWorkspaceCmd->AvailableForStates(G4State_Idle);
  fCudaCudnnConvUseMaxWorkspaceCmd->SetToBeBroadcasted(true);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Par04InferenceMessenger::~Par04InferenceMessenger()
{
  delete fInferenceLibraryCmd;
  delete fSizeLatentVectorCmd;
  delete fSizeConditionVectorCmd;
  delete fModelPathNameCmd;
  delete fModelTypeCmd;
  delete fProfileFlagCmd;
  delete fOptimizationFlagCmd;
  delete fMeshNbRhoCellsCmd;
  delete fMeshNbPhiCellsCmd;
  delete fMeshNbZCellsCmd;
  delete fMeshSizeRhoCellsCmd;
  delete fMeshSizeZCellsCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Par04InferenceMessenger::SetNewValue(G4UIcommand* aCommand, G4String aNewValue)
{
  if (aCommand == fInferenceLibraryCmd) {
    fInference->SetInferenceLibrary(aNewValue);
  }
  if (aCommand == fSizeLatentVectorCmd) {
    fInference->SetSizeLatentVector(std::stoi(aNewValue));
  }
  if (aCommand == fSizeConditionVectorCmd) {
    fInference->SetSizeConditionVector(std::stoi(aNewValue));
  }
  if (aCommand == fModelPathNameCmd) {
    fInference->SetModelPathName(aNewValue);
  }
  if (aCommand == fModelTypeCmd) {
    fInference->SetModelType(aNewValue);
  }
  if (aCommand == fProfileFlagCmd) {
    fInference->SetProfileFlag(std::stoi(aNewValue));
  }
  if (aCommand == fOptimizationFlagCmd) {
    fInference->SetOptimizationFlag(std::stoi(aNewValue));
  }
  else if (aCommand == fMeshNbRhoCellsCmd) {
    fInference->SetMeshNbOfCells(0, fMeshNbRhoCellsCmd->GetNewIntValue(aNewValue));
  }
  else if (aCommand == fMeshNbPhiCellsCmd) {
    fInference->SetMeshNbOfCells(1, fMeshNbPhiCellsCmd->GetNewIntValue(aNewValue));
    fInference->SetMeshSizeOfCells(1,
                                   2. * CLHEP::pi / fMeshNbPhiCellsCmd->GetNewIntValue(aNewValue));
  }
  else if (aCommand == fMeshNbZCellsCmd) {
    fInference->SetMeshNbOfCells(2, fMeshNbZCellsCmd->GetNewIntValue(aNewValue));
  }
  else if (aCommand == fMeshSizeRhoCellsCmd) {
    fInference->SetMeshSizeOfCells(0, fMeshSizeRhoCellsCmd->GetNewDoubleValue(aNewValue));
  }
  else if (aCommand == fMeshSizeZCellsCmd) {
    fInference->SetMeshSizeOfCells(2, fMeshSizeZCellsCmd->GetNewDoubleValue(aNewValue));
  }
  /// Onnx Runtime Execution Provider Flags
  /// Cuda
  if (aCommand == fCudaFlagCmd) {
    fInference->SetCudaFlag(std::stoi(aNewValue));
  }
  if (aCommand == fCudaDeviceIdCmd) {
    fInference->SetCudaDeviceId(aNewValue);
  }
  else if (aCommand == fCudaGpuMemLimitCmd) {
    fInference->SetCudaGpuMemLimit(aNewValue);
  }
  else if (aCommand == fCudaArenaExtendedStrategyCmd) {
    fInference->SetCudaArenaExtendedStrategy(aNewValue);
  }
  else if (aCommand == fCudaCudnnConvAlgoSearchCmd) {
    fInference->SetCudaCudnnConvAlgoSearch(aNewValue);
  }
  else if (aCommand == fCudaDoCopyInDefaultStreamCmd) {
    fInference->SetCudaDoCopyInDefaultStream(aNewValue);
  }
  else if (aCommand == fCudaCudnnConvUseMaxWorkspaceCmd) {
    fInference->SetCudaCudnnConvUseMaxWorkspace(aNewValue);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4String Par04InferenceMessenger::GetCurrentValue(G4UIcommand* aCommand)
{
  G4String cv;

  if (aCommand == fInferenceLibraryCmd) {
    cv = fInferenceLibraryCmd->ConvertToString(fInference->GetInferenceLibrary());
  }
  if (aCommand == fSizeLatentVectorCmd) {
    cv = fSizeLatentVectorCmd->ConvertToString(fInference->GetSizeLatentVector());
  }
  if (aCommand == fSizeConditionVectorCmd) {
    cv = fSizeConditionVectorCmd->ConvertToString(fInference->GetSizeConditionVector());
  }
  if (aCommand == fModelPathNameCmd) {
    cv = fModelPathNameCmd->ConvertToString(fInference->GetModelPathName());
  }
  if (aCommand == fModelTypeCmd) {
    cv = fModelTypeCmd->ConvertToString(fInference->GetModelType());
  }
  if (aCommand == fProfileFlagCmd) {
    cv = fSizeLatentVectorCmd->ConvertToString(fInference->GetProfileFlag());
  }
  if (aCommand == fOptimizationFlagCmd) {
    cv = fSizeLatentVectorCmd->ConvertToString(fInference->GetOptimizationFlag());
  }
  else if (aCommand == fMeshNbRhoCellsCmd) {
    cv = fMeshNbRhoCellsCmd->ConvertToString(fInference->GetMeshNbOfCells()[0]);
  }
  else if (aCommand == fMeshNbPhiCellsCmd) {
    cv = fMeshNbPhiCellsCmd->ConvertToString(fInference->GetMeshNbOfCells()[1]);
  }
  else if (aCommand == fMeshNbZCellsCmd) {
    cv = fMeshNbZCellsCmd->ConvertToString(fInference->GetMeshNbOfCells()[2]);
  }
  else if (aCommand == fMeshSizeRhoCellsCmd) {
    cv = fMeshSizeRhoCellsCmd->ConvertToString(fInference->GetMeshSizeOfCells()[0]);
  }
  else if (aCommand == fMeshSizeZCellsCmd) {
    cv = fMeshSizeZCellsCmd->ConvertToString(fInference->GetMeshSizeOfCells()[2]);
  }
  /// Onnx Runtime Execution Provider Flags
  /// Cuda
  if (aCommand == fCudaDeviceIdCmd) {
    cv = fCudaDeviceIdCmd->ConvertToString(fInference->GetCudaDeviceId());
  }
  else if (aCommand == fCudaGpuMemLimitCmd) {
    cv = fCudaGpuMemLimitCmd->ConvertToString(fInference->GetCudaGpuMemLimit());
  }
  else if (aCommand == fCudaArenaExtendedStrategyCmd) {
    cv = fCudaArenaExtendedStrategyCmd->ConvertToString(fInference->GetCudaArenaExtendedStrategy());
  }
  else if (aCommand == fCudaCudnnConvAlgoSearchCmd) {
    cv = fCudaCudnnConvAlgoSearchCmd->ConvertToString(fInference->GetCudaCudnnConvAlgoSearch());
  }
  else if (aCommand == fCudaDoCopyInDefaultStreamCmd) {
    cv = fCudaDoCopyInDefaultStreamCmd->ConvertToString(fInference->GetCudaDoCopyInDefaultStream());
  }
  else if (aCommand == fCudaCudnnConvUseMaxWorkspaceCmd) {
    cv = fCudaCudnnConvUseMaxWorkspaceCmd->ConvertToString(
      fInference->GetCudaCudnnConvUseMaxWorkspace());
  }

  return cv;
}

#endif
