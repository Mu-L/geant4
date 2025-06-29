# - G4had_preequ_exciton module build definition

# Define the Geant4 Module.
geant4_add_module(G4had_preequ_exciton
  PUBLIC_HEADERS
    G4GNASHTransitions.hh
    G4HETCAlpha.hh
    G4HETCChargedFragment.hh
    G4HETCDeuteron.hh
    G4HETCEmissionFactory.hh
    G4HETCFragment.hh
    G4HETCHe3.hh
    G4HETCNeutron.hh
    G4HETCProton.hh
    G4HETCTriton.hh
    G4LowEGammaNuclearModel.hh
    G4LowEIonFragmentation.hh
    G4PreCompoundAlpha.hh
    G4PreCompoundDeuteron.hh
    G4PreCompoundEmission.hh
    G4PreCompoundEmissionFactory.hh
    G4PreCompoundEmissionInt.hh
    G4PreCompoundFragment.hh
    G4PreCompoundFragmentVector.hh
    G4PreCompoundHe3.hh
    G4PreCompoundInterface.hh
    G4PreCompoundIon.hh
    G4PreCompoundModel.hh
    G4PreCompoundNeutron.hh
    G4PreCompoundNucleon.hh
    G4PreCompoundProton.hh
    G4PreCompoundTransitions.hh
    G4PreCompoundTransitionsInt.hh
    G4PreCompoundTriton.hh
    G4VPreCompoundEmissionFactory.hh
    G4VPreCompoundFragment.hh
    G4VPreCompoundTransitions.hh
  SOURCES
    G4GNASHTransitions.cc
    G4HETCAlpha.cc
    G4HETCChargedFragment.cc
    G4HETCDeuteron.cc
    G4HETCEmissionFactory.cc
    G4HETCFragment.cc
    G4HETCHe3.cc
    G4HETCNeutron.cc
    G4HETCProton.cc
    G4HETCTriton.cc
    G4LowEGammaNuclearModel.cc
    G4LowEIonFragmentation.cc
    G4PreCompoundAlpha.cc
    G4PreCompoundDeuteron.cc
    G4PreCompoundEmission.cc
    G4PreCompoundEmissionFactory.cc
    G4PreCompoundEmissionInt.cc
    G4PreCompoundFragment.cc
    G4PreCompoundFragmentVector.cc
    G4PreCompoundHe3.cc
    G4PreCompoundInterface.cc
    G4PreCompoundIon.cc
    G4PreCompoundModel.cc
    G4PreCompoundNeutron.cc
    G4PreCompoundNucleon.cc
    G4PreCompoundProton.cc
    G4PreCompoundTransitions.cc
    G4PreCompoundTransitionsInt.cc
    G4PreCompoundTriton.cc
    G4VPreCompoundEmissionFactory.cc
    G4VPreCompoundFragment.cc
    G4VPreCompoundTransitions.cc)

geant4_module_link_libraries(G4had_preequ_exciton
  PUBLIC
    G4globman
    G4hadronic_deex_handler
    G4hadronic_mgt
    G4hadronic_util
    G4hepgeometry
    G4hepnumerics
    G4heprandom
    G4partman
  PRIVATE
    G4baryons
    G4hadronic_deex_management
    G4hadronic_deex_util
    G4hadronic_xsect
    G4ions
    G4procman)
