# - G4hadronic_deex_fermi_breakup module build definition

# Define the Geant4 Module.
geant4_add_module(G4hadronic_deex_fermi_breakup
  PUBLIC_HEADERS
    G4FermiBreakUpUtil.hh
    G4FermiBreakUpAN.hh
    G4FermiBreakUpVI.hh
    G4FermiChannels.hh
    G4FermiDataTypes.hh
    G4FermiFragment.hh
    G4FermiFragmentPoolAN.hh
    G4FermiFragmentsPoolVI.hh
    G4FermiIntegerPartition.hh
    G4FermiNucleiProperties.hh
    G4FermiPair.hh
    G4FermiParticle.hh
    G4FermiPhaseDecay.hh
    G4FermiPhaseSpaceDecay.hh
    G4FermiSplitter.hh
    G4FermiStableFragment.hh
    G4FermiUnstableFragment.hh
    G4VFermiBreakUp.hh
    G4VFermiFragmentAN.hh
  SOURCES
    G4FermiBreakUpUtil.cc
    G4FermiBreakUpAN.cc
    G4FermiBreakUpVI.cc
    G4FermiDataTypes.cc
    G4FermiFragment.cc
    G4FermiFragmentPoolAN.cc
    G4FermiFragmentsPoolVI.cc
    G4FermiIntegerPartition.cc
    G4FermiNucleiProperties.cc
    G4FermiPair.cc
    G4FermiParticle.cc
    G4FermiPhaseSpaceDecay.cc
    G4FermiSplitter.cc
    G4FermiStableFragment.cc
    G4FermiUnstableFragment.cc
    G4VFermiFragmentAN.cc)

geant4_module_link_libraries(G4hadronic_deex_fermi_breakup
  PUBLIC
    G4globman
    G4hadronic_util
    G4hepgeometry
    G4heprandom
  PRIVATE
    G4baryons
    G4hadronic_deex_management
    G4partman)
