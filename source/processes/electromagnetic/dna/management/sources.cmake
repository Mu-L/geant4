# - G4emdna-man module build definition

geant4_add_module(G4emdna-man
  PUBLIC_HEADERS
    AddClone_def.hh
    G4AllITFinder.hh
    G4CTCounter.hh
    G4FastList.hh
    G4FastList.icc
    G4IosFlagsSaver.hh
    G4ITBox.hh
    G4IT.hh
    G4ITGun.hh
    G4ITFinder.hh
    G4ITFinder.icc
    G4ITLeadingTracks.hh
    G4ITModelHandler.hh
    G4ITModelManager.hh
    G4ITModelProcessor.hh
    G4ITMultiNavigator.hh
    G4ITNavigator.hh
    G4ITNavigator1.hh
    G4ITNavigator1.icc
    G4ITNavigator2.hh
    G4ITNavigator2.icc
    G4ITPathFinder.hh
    G4ITReaction.hh
    G4ITReactionChange.hh
    G4ITReactionTable.hh
    G4ITSafetyHelper.hh
    G4ITStepProcessor.hh
    G4ITStepStatus.hh
    G4ITSteppingVerbose.hh
    G4ITTrackHolder.hh
    G4ITTrackingInteractivity.hh
    G4ITTrackingManager.hh
    G4ITTransportation.hh
    G4ITTransportation.icc
    G4ITTransportationManager.hh
    G4ITTransportationManager.icc
    G4ITType.hh
    G4ITStepProcessorState_Lock.hh
    G4KDMap.hh
    G4KDNode.hh
    G4KDNode.icc
    G4KDTree.hh
    G4KDTree.icc
    G4KDTreeResult.hh
    G4ManyFastLists.hh
    G4ManyFastLists.icc
    G4MemStat.hh
    G4ReferenceCast.hh
    G4memory.hh
    G4Scheduler.hh
    G4SchedulerMessenger.hh
    G4TrackingInformation.hh
    G4TrackList.hh
    G4TrackState.hh
    G4UserTimeStepAction.hh
    G4VDNAMolecularGeometry.hh
    G4VITDiscreteProcess.hh
    G4VITProcess.hh
    G4VITReactionProcess.hh
    G4VITRestDiscreteProcess.hh
    G4VITRestProcess.hh
    G4VITStepModel.hh
    G4VITSteppingVerbose.hh
    G4VITTimeStepComputer.hh
    G4VITTrackHolder.hh
    G4VScheduler.hh
    G4Octree.hh
    G4Octree.icc
    G4OctreeFinder.hh
    G4OctreeFinder.icc
    G4DNABoundingBox.hh
    G4UserMeshAction.hh
    G4VScavengerMaterial.hh
    G4VUserBrownianAction.hh
    G4VDNAMesh.hh
    G4VUserPulseInfo.hh
    G4MoleculeLocator.hh
  SOURCES
    G4AllITFinder.cc
    G4ITBox.cc
    G4IT.cc
    G4ITGun.cc
    G4ITLeadingTracks.cc
    G4ITModelHandler.cc
    G4ITModelManager.cc
    G4ITModelProcessor.cc
    G4ITMultiNavigator.cc
    G4ITNavigator1.cc
    G4ITNavigator2.cc
    G4ITNavigatorState2.cc
    G4ITPathFinder.cc
    G4ITReaction.cc
    G4ITReactionChange.cc
    G4ITReactionTable.cc
    G4ITSafetyHelper.cc
    G4ITStepProcessor2.cc
    G4ITStepProcessor.cc
    G4ITSteppingVerbose.cc
    G4ITTrackHolder.cc
    G4ITTrackingInteractivity.cc
    G4ITTrackingManager.cc
    G4ITTransportation.cc
    G4ITTransportationManager.cc
    G4ITType.cc
    G4KDMap.cc
    G4KDNode.cc
    G4KDTree.cc
    G4KDTreeResult.cc
    G4MemStat.cc
    G4Scheduler.cc
    G4SchedulerMessenger.cc
    G4TrackingInformation.cc
    G4TrackList.cc
    G4TrackState.cc
    G4UserTimeStepAction.cc
    G4VITDiscreteProcess.cc
    G4VITProcess.cc
    G4VITReactionProcess.cc
    G4VITRestDiscreteProcess.cc
    G4VITRestProcess.cc
    G4VITStepModel.cc
    G4VITSteppingVerbose.cc
    G4VITTimeStepComputer.cc
    G4VITTrackHolder.cc
    G4VScheduler.cc
    G4DNABoundingBox.cc
    G4VUserPulseInfo.cc
    G4MoleculeLocator.cc)

geant4_module_link_libraries(G4emdna-man
  PUBLIC
    G4geometrymng
    G4globman
    G4hepgeometry
    G4heprandom
    G4intercoms
    G4magneticfield
    G4navigation
    G4procman
    G4track
  PRIVATE
    G4cuts
    G4emutils
    G4materials
    G4partman
    G4volumes)
