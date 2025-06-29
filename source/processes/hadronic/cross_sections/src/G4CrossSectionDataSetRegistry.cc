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
// -------------------------------------------------------------------
//
// GEANT4 Class file
//
//
// File name:    G4CrossSectionDataSetRegistry
//
// Author  V.Ivanchenko  24.01.2009
//
// Modifications:
//

#include "G4ios.hh"

#include "G4CrossSectionDataSetRegistry.hh"
#include "G4VCrossSectionDataSet.hh"
#include "G4VComponentCrossSection.hh"
#include "G4CrossSectionFactory.hh"
#include "G4CrossSectionFactoryRegistry.hh"

// Needed for running with 'static' libraries to pull the references of the 
// declared factories
G4_REFERENCE_XS_FACTORY(G4ChipsKaonMinusInelasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsKaonMinusElasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsKaonPlusInelasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsKaonPlusElasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsKaonZeroInelasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsKaonZeroElasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsHyperonInelasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsHyperonElasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsProtonInelasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsProtonElasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsNeutronInelasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsNeutronElasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsPionPlusInelasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsPionPlusElasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsPionMinusInelasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsPionMinusElasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsAntiBaryonInelasticXS);
G4_REFERENCE_XS_FACTORY(G4ChipsAntiBaryonElasticXS);

G4ThreadLocal G4CrossSectionDataSetRegistry* G4CrossSectionDataSetRegistry::instance = nullptr;

G4CrossSectionDataSetRegistry* G4CrossSectionDataSetRegistry::Instance()
{
  if(nullptr == instance) {
    static G4ThreadLocalSingleton<G4CrossSectionDataSetRegistry> inst;
    instance = inst.Instance();
  }
  return instance;
}

G4CrossSectionDataSetRegistry::G4CrossSectionDataSetRegistry()
{}

G4CrossSectionDataSetRegistry::~G4CrossSectionDataSetRegistry()
{
  Clean();
}

void G4CrossSectionDataSetRegistry::Clean()
{
  std::size_t n = xSections.size(); 
  for (std::size_t i=0; i<n; ++i) {
    G4VCrossSectionDataSet* p = xSections[i];
    if(nullptr != p) {
      delete p;
      xSections[i] = nullptr;
    }
  }
  n = xComponents.size(); 
  for (std::size_t i=0; i<n; ++i) {
    auto p = xComponents[i];
    if(nullptr != p) {
      delete p;
      xComponents[i] = nullptr;
    }
  }
}

void G4CrossSectionDataSetRegistry::Register(G4VCrossSectionDataSet* p)
{
  if(nullptr == p) { return; }
  for (auto & xsec : xSections) {
    if(xsec == p) { return; }
  }
  xSections.push_back(p);
}

void G4CrossSectionDataSetRegistry::DeRegister(G4VCrossSectionDataSet* p)
{
  if(nullptr == p) { return; }
  std::size_t n = xSections.size(); 
  for (std::size_t i=0; i<n; ++i) {
    if(xSections[i] == p) {
      xSections[i] = nullptr;
      return;
    }
  }
}

void G4CrossSectionDataSetRegistry::Register(G4VComponentCrossSection* p)
{
  if(nullptr == p) { return; }
  for (auto & xsec : xComponents) {
    if(xsec == p) { return; }
  }
  xComponents.push_back(p);
}

void G4CrossSectionDataSetRegistry::DeRegister(G4VComponentCrossSection* p)
{
  if(nullptr == p) { return; }
  std::size_t n = xComponents.size(); 
  for (std::size_t i=0; i<n; ++i) {
    if(xComponents[i] == p) {
      xComponents[i] = nullptr;
      return;
    }
  }
}

void 
G4CrossSectionDataSetRegistry::DeleteComponent(G4VComponentCrossSection* p)
{
  if(nullptr == p) { return; }
  std::size_t n = xComponents.size(); 
  for (std::size_t i=0; i<n; ++i) {
    if(xComponents[i] == p) {
      delete p;
      return;
    }
  }
}

G4VCrossSectionDataSet* 
G4CrossSectionDataSetRegistry::GetCrossSectionDataSet(const G4String& name, 
                                                      G4bool warning)
{
  for (auto const & xsec : xSections) {
    if (nullptr != xsec && xsec->GetName() == name) { return xsec; }
  }
  // check if factory exists...
  //
  G4CrossSectionFactoryRegistry* factories = 
    G4CrossSectionFactoryRegistry::Instance();
  // This throws if factory is not found, add second parameter 
  // to false to avoid this
  G4VBaseXSFactory* factory = factories->GetFactory(name, warning );
  if (nullptr != factory ) {
    return factory->Instantiate();
  } else {
    G4VCrossSectionDataSet* ptr = nullptr;
    return ptr;
  }
}

G4VComponentCrossSection*
G4CrossSectionDataSetRegistry::GetComponentCrossSection(const G4String& name)
{
  for (auto & xsec : xComponents) {
    if(nullptr != xsec && xsec->GetName() == name) { return xsec; }
  }
  G4VComponentCrossSection* ptr = nullptr;
  return ptr;
}
